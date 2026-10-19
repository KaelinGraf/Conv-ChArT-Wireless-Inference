"""
Test the offline SSL loop (ssl_utils/offline_loop.py): exit conditions, max rounds, resume, split.

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_offline_loop.py -q

The control flow runs with scripted stand-ins for the loop's two GPU stages (relabel_pass and
train_round). The last test runs the whole loop for real on the CPU: the live node's relabel pass
on a rendered board video, with a stand-in for src/inference.py, and RoundRunner against the fake
Conv-ChArT of test_ssl_rounds.py; it needs the Conv-ChArT checkout (dcc).
"""
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
import types

from convchart_ros.ssl_utils.offline_loop import (build_parser, growth_needed, next_step,
                                                  OfflineLoop, open_session)
from convchart_ros.ssl_utils.record import record
import cv2
import numpy as np
import pytest
from ssl_scene import homography, render, result_for
from test_ssl_rounds import FAKE
import yaml


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def make_session(tmp_path, frames=None, **opts):
    """Open a new session over stored frames, with stand-ins for the models and configs.

    Args:
        tmp_path: Folder to build the session in.
        frames: Stamp mapped to image (None writes an empty file), or None for two empty frames.
        **opts: Command-line options of the loop, as keyword arguments.
    """
    stream = tmp_path / 'live' / 'stream'
    stream.mkdir(parents=True)
    for stamp, img in (frames or {1000.0: None, 1000.1: None}).items():
        sec, nsec = int(stamp), round((stamp - int(stamp)) * 10) * 100_000_000
        path = stream / f'{sec:010d}_{nsec:09d}.png'
        if img is None:
            path.write_bytes(b'')
        else:
            cv2.imwrite(str(path), img)
    for name in ('t0.onnx', 'release.onnx'):
        (tmp_path / name).write_bytes(b'')
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text(yaml.safe_dump({'MODEL': {'detector': 'x.onnx', 'refiner': 'r.onnx'},
                                   'CAMERA': {'K': None, 'dist': None}}))
    model = tmp_path / 'model.yaml'
    model.write_text('input_size: [640, 480]\nboard: null\n')
    argv = ['--stream', str(stream), '--out', str(tmp_path / 'session'),
            '--teacher', str(tmp_path / 't0.onnx'), '--base', str(tmp_path / 'release.onnx'),
            '--config', str(cfg), '--model-config', str(model)]
    for k, v in opts.items():
        argv += [f'--{k.replace("_", "-")}', str(v)]
    p, session, resumed = open_session(build_parser().parse_args(argv), tmp_path)
    assert not resumed
    return p, session


class Scripted(OfflineLoop):
    """The offline loop with its GPU stages scripted: frames labelled per pass, accepted per round.

    Args:
        p: Offline loop parameters.
        session: Session folder.
        labelled: Frames labelled by each pass, in pass order.
        accepted: Whether each round's candidate is accepted, in round order.
        resumed: Whether the session is being resumed.
        crash: ('pass', g) to raise in the middle of pass g, or None.
    """

    def __init__(self, p, session, labelled, accepted, resumed=False, crash=None):
        super().__init__(p, session, resumed, log=lambda m: None)
        self.labelled, self.accepted, self.crash, self.calls = labelled, accepted, crash, []

    def relabel_pass(self, g, teacher):
        self.calls.append(('pass', g, teacher))
        (self.session / 'harvest' / f'gen{g}' / 'train' / 'chunk_R0_0000').mkdir(parents=True)
        self.analytics.relabel({'labelled': self.labelled[g - 1]}, g, 0)
        if self.crash == ('pass', g):
            raise RuntimeError('crashed in the middle of a pass')
        n = self.labelled[g - 1]
        return {'kind': 'pass', 'pass': g, 'teacher': teacher, 'labelled': n,
                'train': n - n // 5, 'heldout': n // 5, 'train_dirs': [f'train{g}'],
                'heldout_dirs': [f'heldout{g}']}

    def train_round(self, ps):
        k = len(self.rounds) + 1
        self.calls.append(('round', k, ps['pass'], ps['teacher']))
        return {'kind': 'round', 'round': k, 'pass': ps['pass'], 'teacher': ps['teacher'],
                'candidate_onnx': f'/m/round{k}.onnx', 'accepted': self.accepted[k - 1]}


def test_growth_needed_is_a_share_of_the_previous_pass_and_at_least_one_frame():
    assert growth_needed(1000, 0.005) == 5 and growth_needed(1001, 0.005) == 6
    assert growth_needed(100, 0.005) == 1 and growth_needed(0, 0.005) == 1


def test_next_step_follows_the_exit_conditions():
    def ps(g, n, heldout=1):
        return {'pass': g, 'labelled': n, 'train': n, 'heldout': heldout}

    def rd(k, g, ok, **kw):
        return {'round': k, 'pass': g, 'accepted': ok, **kw}

    assert next_step([], [], 8, 0.005) == ('pass', 1)
    assert next_step([ps(1, 1000)], [], 8, 0.005) == ('round', 1)
    assert next_step([ps(1, 1000)], [rd(1, 1, True)], 8, 0.005) == ('pass', 2)
    assert next_step([ps(1, 1000)], [rd(1, 1, False)], 8, 0.005) == ('exit', 'gate_rejected')
    one = [rd(1, 1, True)]
    assert next_step([ps(1, 1000), ps(2, 1004)], one, 8, 0.005) == ('exit', 'no_new_labels')
    assert next_step([ps(1, 1000), ps(2, 1005)], one, 8, 0.005) == ('round', 2)
    assert next_step([ps(1, 100), ps(2, 100)], one, 8, 0.005) == ('exit', 'no_new_labels')
    assert next_step([ps(1, 100), ps(2, 101)], one, 8, 0.005) == ('round', 2)
    assert next_step([ps(1, 1000)], [rd(1, 1, False, error='stopped')], 8, 0.005) == ('round', 1)
    two = [rd(1, 1, True), rd(2, 2, True)]
    assert next_step([ps(1, 10), ps(2, 20)], two, 2, 0.005) == ('exit', 'max_rounds')
    assert next_step([ps(1, 5, heldout=0)], [], 8, 0.005) == ('exit', 'no_data')


def test_the_loop_stops_when_the_gate_rejects_a_candidate(tmp_path):
    p, s = make_session(tmp_path)
    loop = Scripted(p, s, labelled=[1000, 1100, 1200], accepted=[True, False])
    out = loop.run()
    t0 = p['teacher']
    assert out['exit'] == 'gate_rejected' and out['best_onnx'] == '/m/round1.onnx'
    assert loop.calls == [('pass', 1, t0), ('round', 1, 1, t0), ('pass', 2, '/m/round1.onnx'),
                          ('round', 2, 2, '/m/round1.onnx')]
    lines = jsonl(s / 'loop.jsonl')
    assert [x['kind'] for x in lines] == ['pass', 'round', 'pass', 'round', 'exit']
    assert [x.get('decision') for x in lines[:4]] == ['round 1', 'pass 2', 'round 2',
                                                      'exit: gate_rejected']
    assert lines[2]['growth'] == 100 and lines[2]['growth_needed'] == 5
    summary = json.loads((s / 'summary.json').read_text())
    assert summary['exit'] == 'gate_rejected' and summary['best_is'] == 'round 1'


def test_the_loop_stops_when_a_new_teacher_labels_no_more_frames(tmp_path):
    p, s = make_session(tmp_path)
    loop = Scripted(p, s, labelled=[1000, 1004], accepted=[True])
    out = loop.run()
    assert out['exit'] == 'no_new_labels' and out['best_onnx'] == '/m/round1.onnx'
    assert [c[0] for c in loop.calls] == ['pass', 'round', 'pass']
    last = jsonl(s / 'loop.jsonl')[2]
    assert (last['growth'], last['growth_needed'], last['decision']) == (4, 5,
                                                                         'exit: no_new_labels')


def test_max_rounds_caps_the_loop(tmp_path):
    p, s = make_session(tmp_path, max_rounds=3)
    loop = Scripted(p, s, labelled=[100, 200, 300, 400, 500], accepted=[True] * 5)
    out = loop.run()
    assert out['exit'] == 'max_rounds' and out['best_onnx'] == '/m/round3.onnx'
    assert [c[0] for c in loop.calls] == ['pass', 'round'] * 3 and len(out['rounds']) == 3


def test_a_resumed_loop_redoes_only_the_stage_cut_short(tmp_path):
    p, s = make_session(tmp_path)
    first = Scripted(p, s, labelled=[1000, 1100, 1200], accepted=[True, True, False],
                     crash=('pass', 2))
    with pytest.raises(RuntimeError):
        first.run()
    assert (s / 'harvest' / 'gen2').is_dir() and {r['gen'] for r in jsonl(s / 'relabel.jsonl')} \
        == {1, 2}
    a = build_parser().parse_args(['--resume', str(s), '--max-rounds', '5'])
    p2, s2, resumed = open_session(a, tmp_path)
    assert (s2, resumed, p2['max_rounds'], p2['teacher']) == (s, True, 5, p['teacher'])
    again = Scripted(p2, s2, labelled=[1000, 1100, 1200], accepted=[True, True, False],
                     resumed=True)
    assert not (s / 'harvest' / 'gen2').exists() and (s / 'harvest' / 'gen1').is_dir()
    assert {r['gen'] for r in jsonl(s / 'relabel.jsonl')} == {1}
    out = again.run()
    assert again.calls[0] == ('pass', 2, '/m/round1.onnx')
    assert out['exit'] == 'gate_rejected' and out['best_onnx'] == '/m/round2.onnx'
    assert [x['kind'] for x in jsonl(s / 'loop.jsonl')] == ['pass', 'round', 'pass', 'round',
                                                            'pass', 'round', 'exit']
    meta = json.loads((s / 'session.json').read_text())
    assert meta['resumes'][-1]['params']['max_rounds'] == 5 and meta['params']['max_rounds'] == 8


def test_a_stop_ends_the_loop_without_an_exit_and_resume_retries_the_round(tmp_path):
    class Stopped(Scripted):
        def train_round(self, ps):
            if not self.calls or self.calls[-1][0] != 'stopped':
                self.calls.append(('stopped', ps['pass']))
                self.stop()
                return {'kind': 'round', 'round': 1, 'pass': ps['pass'],
                        'error': 'RuntimeError: round stopped'}
            return super().train_round(ps)

    p, s = make_session(tmp_path)
    assert Stopped(p, s, labelled=[1000], accepted=[False]).run() is None
    assert [x['kind'] for x in jsonl(s / 'loop.jsonl')] == ['pass', 'round']
    assert json.loads((s / 'status.json').read_text())['phase'] == 'stopped'
    assert not (s / 'summary.json').exists()
    p2, s2, _ = open_session(build_parser().parse_args(['--resume', str(s)]), tmp_path)
    again = Scripted(p2, s2, labelled=[1000], accepted=[None, False], resumed=True)
    out = again.run()
    assert again.calls == [('round', 2, 1, p['teacher'])] and out['exit'] == 'gate_rejected'


def key(img):
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()


class Teacher:
    """Stand-in for src/inference.py's pipeline on one detector, over a known rendered video.

    The first teacher and the release find the board every 5th frame of the first stretch only.

    Args:
        table: Frame hash mapped to (frame index, homography).
        detector: Path of the detector ONNX model.
    """

    def __init__(self, table, detector):
        self.table, self.detector = table, detector

    def run_inference(self, img):
        t, H = self.table[key(img)]
        anchor = Path(self.detector).parent.name.startswith('round') or (t < 30 and t % 5 == 0)
        res = result_for(H, anchor=anchor, rng=np.random.default_rng(t))
        res.update(rvec=np.zeros(3) if anchor else None, ambiguous=anchor and t % 4 == 0)
        return res


def test_the_whole_loop_runs_on_the_cpu_and_keeps_the_split_fixed(tmp_path, monkeypatch):
    pytest.importorskip('dcc.video', reason='needs the Conv-ChArT checkout (dcc)')
    rng = np.random.default_rng(0)
    frames, table = {}, {}
    for t in range(60):
        stamp = round((1000.0 if t < 30 else 1007.0) + t / 10, 1)
        img = render(homography(t), noise=1.5, rng=rng)
        frames[stamp], table[key(img)] = img, (t, homography(t))
    fake = types.ModuleType('inference')
    fake.inference_pipeline = lambda cfg: Teacher(
        table, yaml.safe_load(Path(cfg).read_text())['MODEL']['detector'])
    monkeypatch.setitem(sys.modules, 'inference', fake)
    root = tmp_path / 'convchart'
    (root / 'tools').mkdir(parents=True)
    for f in ('video_bootstrap.py', 'train_detector.py'):
        (root / 'tools' / f).write_text('')
    python = tmp_path / 'fake_python'
    python.write_text(FAKE.format(python=sys.executable))
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    p, s = make_session(tmp_path, frames, convchart_root=root)
    p.update(train_python=str(python), split_block_s=1.0, heldout_every=3, chunk_size=10)
    stale = s / 'rounds' / 'round1'
    stale.mkdir(parents=True)
    (stale / 'synth_val.json').write_text('{}')
    out = OfflineLoop(p, s, log=lambda m: None).run()

    assert out['exit'] == 'gate_rejected' and out['best_onnx'] == str(s / 'rounds' / 'round1'
                                                                      / 'detector.onnx')
    lines = jsonl(s / 'loop.jsonl')
    p1, r1, p2, r2, end = lines
    assert p1['labelled'] >= 25 and p2['labelled'] >= p1['labelled'] + 25 and p2['growth'] > 0
    assert p1['heldout'] > 0 and p2['teacher'] == r1['candidate_onnx']
    assert r1['accepted'] and r1['recall']['candidate'] > r1['recall']['teacher']
    assert not r2['accepted'] and not r2['gate']['heldout_recall'] and end['kind'] == 'exit'
    for r in (r1, r2):
        assert all(r['ambiguous_rate'][k] is not None for k in ('base', 'teacher', 'candidate'))
        assert r['candidate']['all']['pose_rate'] > 0
    assert list(s.glob('rounds/round1_cut_*/synth_val.json'))
    assert json.loads((s / 'rounds' / 'round1' / 'synth_val.json').read_text())['m04']
    split = [{x['stamp']: x['split'] for x in jsonl(s / 'harvest' / f'gen{g}' / 'labels.jsonl')}
             for g in (1, 2)]
    assert set(split[0]) < set(split[1]) and all(split[1][k] == v for k, v in split[0].items())
    assert all(v == ('heldout' if math.floor(k) % 3 == 2 else 'train')
               for k, v in split[1].items())
    assert all(Path(x['path']).is_file() for x in jsonl(s / 'harvest' / 'gen2' / 'labels.jsonl'))
    pieces = [[x['piece'] for x in ps['pieces']] for ps in (p1, p2)]
    assert pieces[0] == pieces[1] == [str(s / 'pieces' / f'R0p{k}') for k in range(4)]
    text = record(s, tmp_path / 'record').read_text()
    assert '## Loop' in text and '- Ended: gate_rejected' in text and 'Offline looped' in text
