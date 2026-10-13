"""
Run self-supervised fine-tuning rounds in a loop over camera frames already on disk (no ROS).

    python3 -m convchart_ros.ssl_utils.offline_loop --stream <live session>/stream \
        [--stream DIR ...] [--out ssl_output/offline_<stamp>] [--teacher ONNX] [--base ONNX] \
        [--max-rounds 8] [--min-growth 0.005] [--record-dir DIR]
    python3 -m convchart_ros.ssl_utils.offline_loop --resume <session folder>|latest

The live node's own machinery in a loop: SSLNode's relabel pass, teacher pipelines and held-out
evaluation (borrowed by Headless below; no node is started) and ssl_utils.rounds.RoundRunner,
with the node's defaults (ssl_node.DEFAULTS).

1. Pass g: the teacher relabels every stored frame, oldest first, into the bank harvest/gen<g>/:
   anchors where its identified corners fit the lattice, tracking between them, every labelled
   frame kept; a frame is held out when its capture time falls in every 5th block of 30 s, so it
   never changes split. The first teacher is the last accepted model of the live session the
   first stream was recorded in, unless --teacher names one. Each stream is relabelled as
   --relabel-workers contiguous pieces at once (the node's relabel threads per piece, one shared
   teacher), since labelling, not the GPU, sets the pace; the pieces are the same in every pass,
   and tracking does not cross their (few) ends, where it is one-sided instead.
2. Round: a candidate is trained from the base weights (the release detector) on every train
   frame of gen<g>, 40% of each batch synthetic and every real frame drawn at least 8 times, and
   is gated against the teacher on gen<g>'s held-out frames.
3. A rejected candidate ends the loop (validation did not improve). An accepted one becomes the
   teacher and the next pass relabels every frame with it. If that pass labels fewer than
   max(1, min_growth x the previous pass's count) more frames, the loop ends (no more frames can
   be labelled); otherwise back to 2. At most max_rounds rounds.

The session folder holds loop.jsonl (one line per pass and per round: teacher, counts, growth,
gate metrics with far-board recall and the share of held-out poses src/inference.py finds
IPPE-ambiguous for the base, the teacher and the candidate, the decision, timings), status.json
(progress and ETA), harvest/gen<g>/ (the labelled frames; labels.jsonl names each label's frame
file), rounds/round<k>/ (detector.onnx, round.log, ...), rounds.jsonl, relabel.jsonl and, at the
end, summary.json (why the loop ended, the best model). A stage is logged once it is complete,
so --resume redoes at most the one that was cut short. SIGTERM or SIGINT stops the loop cleanly
and terminates a running training subprocess. With --record-dir, ssl_utils.record rewrites
<record-dir>/<session>/ after every stage. No model is deployed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import yaml

from .analytics import _read_jsonl, Analytics
from .harvester import HarvestParams, OnlineHarvester
from .record import _banked, LEVELS, record, SCALES
from .resume import generations, read_rounds, resume_state
from .rounds import evaluate_onnx, RoundConfig, RoundRunner
from ..ssl_node import DEFAULTS, Generation, RELEASE_DETECTOR, SSLNode, stamp_of, stored_frames

MODEL_CONFIG = Path('configs') / 'rel_w882_c2.yaml'
BACKGROUNDS = Path('runs') / 'vb_test' / 'bgs'
NODE_KEYS = ('passes', 'min_steps', 'batch', 'accum', 'lr', 'real_frac', 'darken_synthetic',
             'workers', 'val_size', 'train_python', 'max_gap', 'max_dt_s', 'min_motion_px',
             'cap_per_bucket', 'chunk_size', 'heldout_every', 'split_block_s')
LOOP_DEFAULTS = {'max_rounds': 8, 'min_growth': 0.005, 'record_dir': '', 'relabel_workers': 4}
RESUMABLE = ('max_rounds', 'min_growth', 'record_dir')
OPTIONS = RESUMABLE + ('passes', 'min_steps', 'workers', 'val_size', 'relabel_workers')


def find_repo() -> Path:
    """Return the wireless-inference checkout this file belongs to (else the working folder)."""
    for d in Path(__file__).resolve().parents:
        if (d / 'cfg' / 'cfg.yaml').is_file() and (d / 'checkpoints').is_dir():
            return d
    return Path.cwd()


def growth_needed(prev: int, min_growth: float) -> int:
    """Return how many more frames than prev a pass must label for the loop to carry on."""
    return max(1, math.ceil(min_growth * prev - 1e-9))


def next_step(passes: list, rounds: list, max_rounds: int, min_growth: float) -> tuple:
    """Decide what the loop does next from the loop.jsonl lines of its passes and rounds."""
    if not passes:
        return ('pass', 1)
    last = passes[-1]
    scored = [r for r in rounds if 'error' not in r]
    done = [r for r in scored if r['pass'] == last['pass']]
    if not done:
        if len(passes) > 1:
            prev = passes[-2]['labelled']
            if last['labelled'] - prev < growth_needed(prev, min_growth):
                return ('exit', 'no_new_labels')
        if not last['train'] or not last['heldout']:
            return ('exit', 'no_data')
        if len(scored) >= max_rounds:
            return ('exit', 'max_rounds')
        return ('round', last['pass'])
    if not done[-1]['accepted']:
        return ('exit', 'gate_rejected')
    if len(scored) >= max_rounds:
        return ('exit', 'max_rounds')
    return ('pass', last['pass'] + 1)


def first_teacher(stream: str) -> str:
    """Return the last accepted model of the live session that recorded a stream folder."""
    session = Path(stream).parent
    best = resume_state(session)['best_onnx'] if (session / 'session.json').is_file() else None
    if best:
        return best
    raise SystemExit(f'{session} has no accepted model: name the first teacher with --teacher')


def _hms(s) -> str:
    if s is None:
        return '-'
    s = int(max(0.0, s))
    return f'{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}'


def _group(ev: dict | None, group: str, key: str):
    return ((ev or {}).get(group) or {}).get(key)


class Headless:
    """Stand in for an SSLNode where its relabel pass, pipelines and evaluation need one."""

    _model_path = SSLNode._model_path
    _pipeline_for = SSLNode._pipeline_for
    _evaluate = SSLNode._evaluate
    _harvester_for = SSLNode._harvester_for
    _teach = SSLNode._teach
    _relabel_teacher = SSLNode._relabel_teacher
    _relabel_harvest = SSLNode._relabel_harvest

    def __init__(self, p: dict, repo: Path, session: Path, base_cfg: dict, model_cfg: dict,
                 analytics: Analytics, log):
        import inference
        self.p, self.repo, self.session = p, repo, session
        self.base_cfg, self.model_cfg = base_cfg, model_cfg
        self.analytics, self._log = analytics, log
        self._inference, self._evaluate_onnx = inference, evaluate_onnx
        self._Harvester, self._HarvestParams = OnlineHarvester, HarvestParams
        self.refiner_onnx = self._model_path('refiner')
        self.harvester = self._harvester_for(0, '')
        self.teacher = self.teacher_onnx = self.stream_dir = None
        self._teacher_lock, self._stop = threading.Lock(), threading.Event()
        self._frames, self._to_save = deque(), deque()

    def get_logger(self):
        return self

    def info(self, msg, **_):
        self._log(msg)

    error = warn = info

    def _maybe_round(self) -> None:
        pass


class OfflineLoop:
    """The loop over one session folder; relabel_pass and train_round do the GPU work."""

    def __init__(self, p: dict, session: Path, resumed: bool = False, log=None):
        self.p, self.session, self.resumed = p, Path(session), resumed
        self.log = log or (lambda m: print(f'[{datetime.now():%H:%M:%S}] {m}', flush=True))
        self.repo = find_repo()
        self.streams = [Path(d) for d in p['streams']]
        meta = json.loads((self.session / 'session.json').read_text())
        self.base_cfg = yaml.safe_load(Path(meta['config_snapshot']).read_text())
        self.model_cfg = yaml.safe_load(Path(p['model_config']).read_text())
        lines = _read_jsonl(self.session / 'loop.jsonl')
        self.passes = [x for x in lines if x.get('kind') == 'pass']
        self.rounds = [x for x in lines if x.get('kind') == 'round']
        self.t_start = meta.get('t_start', time.time())
        if resumed:
            self._drop_unfinished_passes()
        self.analytics = Analytics(self.session, resume=True)
        self._node = self.rr = None
        self.extra, self.gens = [], []
        self.stopping = False

    @property
    def node(self) -> Headless:
        if self._node is None:
            for d in (self.repo / 'src', Path(self.p['convchart_root'])):
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
            self._node = Headless(self.p, self.repo, self.session, self.base_cfg,
                                  self.model_cfg, self.analytics, self.log)
        return self._node

    def piece_node(self, k: int) -> Headless:
        """Return the node that relabels piece k of a stream (piece 0: the main one)."""
        while len(self.extra) < k:
            self.extra.append(Headless(self.p, self.repo, self.session, self.base_cfg,
                                       self.model_cfg, self.analytics, self.log))
        return self.node if k == 0 else self.extra[k - 1]

    def run(self) -> dict | None:
        """Run passes and rounds until an exit condition; return the summary (None if stopped)."""
        self.log(f'offline loop in {self.session}' + (' (resumed)' if self.resumed else '')
                 + f': {len(self.passes)} passes and {len(self.rounds)} rounds done')
        while not self.stopping:
            step = next_step(self.passes, self.rounds, self.p['max_rounds'],
                             self.p['min_growth'])
            if step[0] == 'exit':
                return self.finish(step[1])
            if step[0] == 'pass':
                g = step[1]
                teacher = self.p['teacher'] if g == 1 else next(
                    r['candidate_onnx'] for r in self.rounds
                    if r['pass'] == g - 1 and r.get('accepted'))
                line = self.relabel_pass(g, teacher)
                if line is None:
                    break
                if self.passes:
                    prev = self.passes[-1]['labelled']
                    line.update(growth=line['labelled'] - prev,
                                growth_needed=growth_needed(prev, self.p['min_growth']))
                self.passes.append(line)
            else:
                line = self.train_round(self.passes[-1])
                if line is None:
                    break
                self.rounds.append(line)
            step = next_step(self.passes, self.rounds, self.p['max_rounds'], self.p['min_growth'])
            line['decision'] = ('exit: error' if 'error' in line and not self.stopping
                                else ' '.join(map(str, step)).replace('exit ', 'exit: '))
            self._write(line)
            self._record()
            if 'error' in line and not self.stopping:
                return self.finish('error', line['error'])
        self.log(f'stopped; carry on with --resume {self.session}')
        self._status(phase='stopped')
        return None

    def stop(self, *_) -> None:
        """End the loop soon: cancel the relabel pass, terminate a running round."""
        self.stopping = True
        for node in [self._node, *self.extra]:
            if node is not None:
                node._stop.set()
        for gen in self.gens:
            gen.cancel.set()
        if self.rr is not None:
            self.rr.stop()

    def relabel_pass(self, g: int, teacher: str) -> dict | None:
        """Relabel every stored frame with teacher into harvest/gen<g>/ (None if stopped)."""
        t0 = time.time()
        self.log(f'pass {g}: relabelling with {teacher}')
        pipe, lock = self.node._pipeline_for(teacher), threading.Lock()
        snaps, done = [], 0
        try:
            for i, d in enumerate(self.streams):
                pieces = self._pieces(i, d)
                self.gens, threads = [], []
                for k, piece in enumerate(pieces):
                    node = self.piece_node(k)
                    node.teacher, node.teacher_onnx, node._teacher_lock = pipe, teacher, lock
                    node.stream_dir = piece
                    self.gens.append(Generation(g, teacher, stamp_of(stored_frames(piece)[-1]),
                                                None, node._harvester_for(g, f'R{i}p{k}_')))
                    threads += [threading.Thread(target=f, args=(self.gens[-1],), daemon=True)
                                for f in (node._relabel_teacher, node._relabel_harvest)]
                for t in threads:
                    t.start()
                self._watch_pass(g, i, threads)
                if not all(gen.finished for gen in self.gens):
                    return None
                for gen, piece in zip(self.gens, pieces):
                    done += gen.done
                    snaps.append({'stream': str(d), 'piece': str(piece), 'frames': gen.done,
                                  **gen.relabeler.snapshot()})
        finally:
            for node in [self._node, *self.extra]:
                node.teacher = None
            del pipe
            gc.collect()
        totals = Counter()
        for s in snaps:
            totals.update(s['totals'])
        chunks = {k: sorted(c for s in snaps for c in s['chunks'][k])
                  for k in ('train', 'heldout')}
        bank = _banked(self.session, g)
        n = {k: bank[k]['frames'] for k in ('train', 'heldout')}
        self._write_labels(g, chunks)
        secs = time.time() - t0
        self.log(f'pass {g}: {n["train"]} train / {n["heldout"]} held-out frames labelled from '
                 f'{done} in {secs / 60:.1f} min')
        return {'kind': 'pass', 'pass': g, 'teacher': teacher, 'frames': done,
                'labelled': n['train'] + n['heldout'], 'train': n['train'],
                'heldout': n['heldout'], 'detected_anchors': totals['anchors'],
                'anchor': totals['anchor'], 'tracked': totals['tracked'],
                'breaks': totals['breaks'],
                'sources': {k: bank[k]['sources'] for k in ('train', 'heldout')},
                'by_size_and_brightness': {'rows': SCALES, 'cols': LEVELS,
                                           **{k: bank[k]['grid'].tolist()
                                              for k in ('train', 'heldout')}},
                'oblique': {k: bank[k]['oblique'] for k in ('train', 'heldout')},
                'pieces': [{k: s[k] for k in ('stream', 'piece', 'frames')} for s in snaps],
                'train_dirs': chunks['train'], 'heldout_dirs': chunks['heldout'],
                't_start': t0, 'seconds': round(secs, 1),
                'fps': round(done / secs, 2) if secs > 0 else None}

    def _pieces(self, i: int, d: Path) -> list:
        """Return stream i as relabel_workers folders of links to contiguous runs of its frames."""
        names = stored_frames(d)
        k = int(self.p.get('relabel_workers') or 1)
        if not names or k < 2 or len(names) < 2 * k:
            return [d] if names else []
        out = []
        for j in range(k):
            piece = self.session / 'pieces' / f'R{i}p{j}'
            if not piece.is_dir():
                tmp = piece.with_name(piece.name + '.tmp')
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                for n in names[j * len(names) // k:(j + 1) * len(names) // k]:
                    (tmp / n).symlink_to(d / n)
                tmp.rename(piece)
            out.append(piece)
        return out

    def _watch_pass(self, g: int, i: int, threads: list) -> None:
        last, start = 0.0, time.time()
        while any(t.is_alive() for t in threads):
            next(t for t in threads if t.is_alive()).join(timeout=2.0)
            if time.time() - last < 30.0 and any(t.is_alive() for t in threads):
                continue
            last = time.time()
            done, total = sum(x.done for x in self.gens), sum(x.total for x in self.gens)
            fps = round(done / (last - start), 1) if last > start else 0.0
            eta = (total - done) / fps if fps else None
            labelled = sum(x.relabeler.snapshot()['totals'].get('labelled', 0) for x in self.gens)
            self._status(**{'phase': 'relabel', 'pass': g, 'stream': i, 'done': done,
                            'total': total, 'fps': fps, 'labelled': labelled, 'eta_s': eta})
            self.log(f'pass {g}: {done}/{total} frames at {fps}/s, {labelled} labelled, '
                     f'ETA {_hms(eta)}')

    def _write_labels(self, g: int, chunks: dict) -> None:
        """Name each label's frame file: harvest/gen<g>/labels.jsonl."""
        path_of = {stamp_of(n): str(d / n) for d in self.streams for n in stored_frames(d)}
        bank = self.session / 'harvest' / f'gen{g}'
        bank.mkdir(parents=True, exist_ok=True)
        with open(bank / 'labels.jsonl', 'w') as f:
            for split in ('train', 'heldout'):
                for c in chunks[split]:
                    recs = json.loads((Path(c) / 'records.json').read_text())['records']
                    for j, r in enumerate(recs):
                        f.write(json.dumps({
                            'path': path_of.get(r['stamp']), 'stamp': r['stamp'], 'split': split,
                            'chunk': Path(c).name, 'index': j, 'source': r['source'],
                            's_px': round(r['s_px'], 2),
                            'board_level': round(r['board_level'], 1)}) + '\n')

    def _drop_unfinished_passes(self) -> None:
        """On resume, delete the bank and relabel.jsonl rows of a pass cut short."""
        done = {x['pass'] for x in self.passes}
        for g in generations(self.session):
            if g not in done:
                shutil.rmtree(self.session / 'harvest' / f'gen{g}', ignore_errors=True)
                self.log(f'pass {g} was cut short: it starts again')
        path = self.session / 'relabel.jsonl'
        if path.exists():
            rows = _read_jsonl(path)
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows if r.get('gen') in done))

    def round_config(self) -> RoundConfig:
        p = self.p
        return RoundConfig(
            convchart_root=p['convchart_root'], python=p['train_python'] or sys.executable,
            base_onnx=p['base_detector'], model_config=p['model_config'],
            workdir=str(self.session / 'rounds'), backgrounds=p['backgrounds'],
            passes=float(p['passes']), min_steps=int(p['min_steps']), batch=p['batch'],
            accum=p['accum'], lr=p['lr'], real_frac=float(p['real_frac']),
            darken_synthetic=p['darken_synthetic'], workers=p['workers'],
            val_size=p['val_size'], val_subset=min(150, p['val_size']))

    def train_round(self, ps: dict) -> dict | None:
        """Train a candidate on pass ps's bank, gate it against ps's teacher (None if stopped)."""
        node = self.node
        rr = RoundRunner(self.round_config(), node._evaluate, log=self.log)
        rr.restore(read_rounds(self.session), ps['teacher'])
        rd = self.session / 'rounds' / f'round{rr.round + 1}'
        if rd.exists():
            rd.rename(rd.with_name(f'{rd.name}_cut_{datetime.now():%Y%m%d-%H%M%S}'))
        self.rr = rr
        if self.stopping or not rr.start(ps['train_dirs'], ps['heldout_dirs']):
            return None
        self.log(f'round {rr.round} on pass {ps["pass"]}: {ps["train"]} train / '
                 f'{ps["heldout"]} held-out frames, {rr.steps or "?"} steps')
        self._watch_round(rr)
        res = rr.status()['history'][-1]
        self.analytics.round_done(res)
        base = None
        if 'candidate' in res and not self.stopping:
            base = node._evaluate(self.p['base_detector'], ps['heldout_dirs'])
        ref, cand = res.get('reference'), res.get('candidate')
        line = {'kind': 'round', 'round': res['round'], 'pass': ps['pass'],
                'teacher': ps['teacher'], 'candidate_onnx': res.get('candidate_onnx'),
                'accepted': bool(res.get('accepted')), 'gate': res.get('gate'),
                'train_frames': res['train_frames'], 'heldout_frames': res['heldout_frames'],
                'steps': res['steps'], 'passes': res['passes'], 'run': res.get('run'),
                'recall': {'teacher': _group(ref, 'all', 'recall'),
                           'candidate': _group(cand, 'all', 'recall')},
                'far_recall': {'teacher': _group(ref, 'far <16 px', 'recall'),
                               'candidate': _group(cand, 'far <16 px', 'recall')},
                'ambiguous_rate': {k: _group(ev, 'all', 'ambiguous_rate')
                                   for k, ev in (('base', base), ('teacher', ref),
                                                 ('candidate', cand))},
                'base': base, 'reference': ref, 'candidate': cand,
                'synth_base': res.get('synth_base'), 'synth_candidate': res.get('synth_candidate'),
                't_start': res['t_start'], 'seconds': res.get('seconds')}
        if 'error' in res:
            line['error'] = res['error']
        self.log(f'round {line["round"]}: accepted={line["accepted"]} gate={line["gate"]} '
                 f'recall {line["recall"]} far {line["far_recall"]} ambiguous '
                 f'{line["ambiguous_rate"]}'
                 + (f' error {line["error"]}' if 'error' in line else ''))
        return line

    def _watch_round(self, rr: RoundRunner) -> None:
        last, start = 0.0, None
        while rr.thread.is_alive():
            rr.thread.join(timeout=2.0)
            st, now = rr.status(), time.time()
            if st['phase'] == 'train' and start is None:
                start = (now, st['progress'])
            if now - last < 60.0:
                continue
            last = now
            rate = ((st['progress'] - start[1]) / (now - start[0])
                    if start and now > start[0] else 0.0)
            eta = (1.0 - st['progress']) / rate if st['phase'] == 'train' and rate > 0 else None
            self._status(phase='round', round=st['round'], round_phase=st['phase'],
                         progress=st['progress'], steps=rr.steps, train_eta_s=eta)
            self.log(f'round {st["round"]}: {st["phase"]} {100 * st["progress"]:.0f}% of '
                     f'{rr.steps} steps, training ETA {_hms(eta)}')

    def finish(self, reason: str, detail: str | None = None) -> dict:
        """Write summary.json and the exit line; return the summary."""
        accepted = [r for r in self.rounds if r.get('accepted')]
        best = accepted[-1]['candidate_onnx'] if accepted else self.p['teacher']
        summary = {
            'exit': reason, 'detail': detail, 'best_onnx': best,
            'best_is': f'round {accepted[-1]["round"]}' if accepted else 'the first teacher',
            'first_teacher': self.p['teacher'], 'base': self.p['base_detector'],
            'passes': [{k: x.get(k) for k in ('pass', 'teacher', 'frames', 'labelled', 'train',
                                              'heldout', 'growth', 'growth_needed', 'seconds')}
                       for x in self.passes],
            'rounds': [{k: x.get(k) for k in ('round', 'pass', 'accepted', 'gate', 'recall',
                                              'far_recall', 'ambiguous_rate', 'steps', 'seconds',
                                              'candidate_onnx', 'error')}
                       for x in self.rounds],
            't_start': self.t_start, 't_end': time.time(),
            'hours': round((time.time() - self.t_start) / 3600.0, 2)}
        (self.session / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        self._write({'kind': 'exit', 'reason': reason, 'detail': detail, 'best_onnx': best,
                     't': time.time()})
        self._status(phase='done', exit=reason, best_onnx=best)
        self._record()
        self.log(f'done: {reason}{f" ({detail})" if detail else ""}; best model {best}')
        return summary

    def _write(self, line: dict) -> None:
        with open(self.session / 'loop.jsonl', 'a') as f:
            f.write(json.dumps(line, default=str) + '\n')

    def _status(self, **kw) -> None:
        tmp = self.session / 'status.json.tmp'
        tmp.write_text(json.dumps({**kw, 'updated': datetime.now().isoformat(timespec='seconds')},
                                  default=str))
        tmp.replace(self.session / 'status.json')

    def _record(self) -> None:
        if not self.p.get('record_dir'):
            return
        try:
            record(self.session, Path(self.p['record_dir']) / self.session.name)
        except Exception as e:
            self.log(f'record failed: {type(e).__name__}: {e}')


def open_session(a: argparse.Namespace, repo: Path) -> tuple:
    """Create or resume the session folder a asks for; return (params, folder, resumed)."""
    if a.resume:
        root = repo / 'ssl_output'
        session = (max((d for d in root.glob('offline_*') if (d / 'session.json').exists()),
                       default=None)
                   if a.resume == 'latest' else Path(a.resume).expanduser().resolve())
        if session is None or not (session / 'session.json').exists():
            raise SystemExit(f'--resume: no offline session at {a.resume!r}')
        meta = json.loads((session / 'session.json').read_text())
        p = dict((meta.get('resumes') or [{}])[-1].get('params') or meta['params'])
        p.update({k: getattr(a, k) for k in RESUMABLE if getattr(a, k) is not None})
        meta.setdefault('resumes', []).append(
            {'at': datetime.now().isoformat(timespec='seconds'), 'argv': sys.argv, 'params': p})
        (session / 'session.json').write_text(json.dumps(meta, indent=1))
        return p, session, True
    if not a.stream:
        raise SystemExit('name the frames with --stream <folder> (or --resume a session)')
    convchart = Path(a.convchart_root or repo.parent / 'Conv-ChArT').resolve()
    p = {**{k: DEFAULTS[k] for k in NODE_KEYS}, **LOOP_DEFAULTS,
         'streams': [str(Path(s).resolve()) for s in a.stream],
         'config': str(Path(a.config or repo / 'cfg' / 'cfg.yaml').resolve()),
         'model_config': str(Path(a.model_config or convchart / MODEL_CONFIG).resolve()),
         'convchart_root': str(convchart),
         'backgrounds': str(Path(a.backgrounds or convchart / BACKGROUNDS).resolve()),
         'base_detector': str(Path(a.base or repo / RELEASE_DETECTOR).resolve())}
    p.update({k: getattr(a, k) for k in OPTIONS if getattr(a, k) is not None})
    p['teacher'] = str(Path(a.teacher).resolve()) if a.teacher else first_teacher(p['streams'][0])
    for k in ('teacher', 'base_detector', 'config', 'model_config'):
        if not Path(p[k]).is_file():
            raise SystemExit(f'{k}: {p[k]} does not exist')
    session = Path(a.out).resolve() if a.out else (
        repo / 'ssl_output' / f'offline_{datetime.now():%Y%m%d-%H%M%S}')
    if (session / 'session.json').exists():
        raise SystemExit(f'{session} is already a session: carry it on with --resume')
    (session / 'cfg').mkdir(parents=True, exist_ok=True)
    snapshot = session / 'cfg' / 'cfg.yaml'
    shutil.copy2(p['config'], snapshot)
    cam = (yaml.safe_load(snapshot.read_text()) or {}).get('CAMERA') or {}
    streams = []
    for d in p['streams']:
        names = stored_frames(Path(d))
        streams.append({'dir': d, 'frames': len(names),
                        'first': stamp_of(names[0]) if names else None,
                        'last': stamp_of(names[-1]) if names else None})
    meta = {'kind': 'offline', 'params': p, 'config': p['config'],
            'config_snapshot': str(snapshot), 'convchart_root': p['convchart_root'],
            'detector': p['base_detector'], 'base_detector': p['base_detector'],
            'teacher': p['teacher'], 'streams': streams,
            'lens': 'calibrated' if cam.get('K') is not None else 'uncalibrated',
            'created': datetime.now().isoformat(timespec='seconds'), 't_start': time.time(),
            'argv': sys.argv}
    (session / 'session.json').write_text(json.dumps(meta, indent=1))
    return p, session, False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stream', action='append', default=[],
                    help='a folder of frames named <sec>_<nanosec>.png (repeat for more)')
    ap.add_argument('--out', default='', help='session folder (ssl_output/offline_<stamp>)')
    ap.add_argument('--resume', default='', help='carry on in this session folder, or latest')
    ap.add_argument('--teacher', default='',
                    help="first teacher (the last accepted model of the first stream's session)")
    ap.add_argument('--base', default='', help='weights every round trains from (the release)')
    ap.add_argument('--config', default='', help='pipeline config (cfg/cfg.yaml): its CAMERA, '
                    'PIPELINE, BOARD and MODEL.refiner are used')
    ap.add_argument('--model-config', default='', help='Conv-ChArT config of the detector')
    ap.add_argument('--convchart-root', default='', help='Conv-ChArT checkout (../Conv-ChArT)')
    ap.add_argument('--backgrounds', default='', help='images for the synthetic share')
    ap.add_argument('--max-rounds', type=int, default=None, help='at most this many rounds (8)')
    ap.add_argument('--min-growth', type=float, default=None,
                    help='stop unless a new teacher labels this share more frames (0.005)')
    ap.add_argument('--record-dir', default=None,
                    help='rewrite ssl_utils.record into <dir>/<session>/ after every stage')
    ap.add_argument('--passes', type=float, default=None, help='draws per real frame (8)')
    ap.add_argument('--min-steps', type=int, default=None, help='shortest round (0)')
    ap.add_argument('--workers', type=int, default=None, help='training loader workers (8)')
    ap.add_argument('--val-size', type=int, default=None, help='synthetic validation size (300)')
    ap.add_argument('--relabel-workers', type=int, default=None,
                    help='pieces of each stream relabelled at once (4)')
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    p, session, resumed = open_session(a, find_repo())
    loop = OfflineLoop(p, session, resumed)
    signal.signal(signal.SIGTERM, loop.stop)
    signal.signal(signal.SIGINT, loop.stop)
    loop.run()


if __name__ == '__main__':
    main()
