"""
A self-contained record of one ssl fine-tune session, for evaluating it and for the report (ROS-free).

    python3 -m convchart_ros.ssl_utils.record <ssl_output/session> <out_dir>

A live session (ssl_node), or an offline one (ssl_utils.offline_loop; its passes and rounds are
in loop.jsonl).

Writes <out_dir>/README.md: what ran and with which settings, the data the session gathered, each round's training
and gate results, and the caveats that matter when the numbers are evaluated. Copies everything small next to it:
data/ (session.json, rounds.jsonl, segments.jsonl, base_synth_val.json, and per round result.json, synth_val.json,
finetune.yaml and the training metrics.jsonl; summary.json once the session has ended) and figures/ (dashboard.png;
report/*.pdf once the session has ended). The frames (harvest/, stream/), the candidate models and the checkpoints
stay where they are; the README names them. Re-run it as rounds finish: it rewrites the same files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .resume import generations

SCALES = ["<16 px", "16-32 px", "32-64 px", ">=64 px"]
LEVELS = ["dark", "dim", "bright"]
try:
    from zoneinfo import ZoneInfo
    ZONE = ZoneInfo("Pacific/Auckland")
except Exception:                                   # noqa: BLE001
    ZONE = None


def _when(t=None) -> str:
    d = datetime.fromtimestamp(t, ZONE) if t is not None else datetime.now(ZONE)
    return d.strftime("%Y-%m-%d %H:%M:%S %Z") if ZONE else d.strftime("%Y-%m-%d %H:%M:%S UTC")


def _local(path, root: Path) -> Path:
    """Return a path as logged (inside the container, under /workspace) as it is on this machine.

    Args:
        path: Path as logged.
        root: Local root of the repo, standing in for /workspace.
    """
    p = Path(str(path))
    if p.exists() or not str(p).startswith("/workspace/"):
        return p
    return root / p.relative_to("/workspace")


def _jsonl(path: Path) -> list:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _bucket(rec) -> tuple:
    s, lv = rec["s_px"], rec["board_level"]
    return (0 if s < 16 else 1 if s < 32 else 2 if s < 64 else 3), (0 if lv < 20 else 1 if lv < 60 else 2)


def _chunk_stats(chunks) -> dict:
    """Return frames, label sources, labelled corners and coverage of some banked chunks.

    Args:
        chunks: Chunk folders, each with images.npy and records.json.
    """
    recs = []
    for c in chunks:
        f = Path(c) / "records.json"
        if f.exists():
            recs += json.loads(f.read_text()).get("records", [])
    grid = np.zeros((4, 3), int)
    for r in recs:
        s, b = _bucket(r)
        grid[s, b] += 1
    return {"chunks": [Path(c).name for c in chunks], "frames": len(recs), "grid": grid,
            "oblique": sum(int(r.get("anisotropy", 0.0) > 0.15) for r in recs),
            "corners": sum(sum(1 for c in r.get("corners", []) if c.get("visible") is True) for r in recs),
            "sources": {k: sum(r.get("source") == k for r in recs) for k in sorted({r.get("source") for r in recs})}}


def _bank_dir(session: Path, gen: int) -> Path:
    return session / "harvest" if gen == 0 else session / "harvest" / f"gen{gen}"


def _banked(session: Path, gen: int = 0) -> dict:
    """Return a bank generation's frame statistics, per split.

    Args:
        session: Session folder.
        gen: Bank generation; 0 is a session's bank from before generations.
    """
    root = _bank_dir(session, gen)
    return {split: _chunk_stats(sorted((root / split).glob("chunk_*"))) for split in ("train", "heldout")}


def _losses(metrics: Path) -> dict:
    rows = _jsonl(metrics)
    steps = [r for r in rows if "loss" in r]
    vals = [r for r in rows if "val" in r]
    done = next((r["done"] for r in rows if "done" in r), {})
    loss = [r["loss"] for r in steps]
    return {"first50": float(np.mean(loss[:50])) if loss else None,
            "last50": float(np.mean(loss[-50:])) if loss else None,
            "steps": len(steps), "val": [(r["step"], r["val"].get("val_loss"),
                                          (r["val"].get("m04") or {}).get("accuracy")) for r in vals],
            "train_s": done.get("train_elapsed_s"), "samples_per_s": done.get("samples_per_s")}


def _f(x, fmt="{:.4f}"):
    return "-" if x is None else fmt.format(x)


def _offline_what(meta: dict, p: dict, loop: list, n_bgs: int) -> list:
    """Return the "What ran" lines of an offline session (ssl_utils.offline_loop).

    Args:
        meta: Contents of session.json.
        p: Offline loop parameters.
        loop: Lines of loop.jsonl.
        n_bgs: Number of background images.
    """
    out = ["- Offline looped self-supervised fine-tuning of the Conv-ChArT detector over frames "
           "already on disk (`python3 -m convchart_ros.ssl_utils.offline_loop`): each pass "
           "relabels every stored frame with the teacher; a candidate trained from the base on "
           "all of the pass's train frames is gated against the teacher on its held-out frames; "
           "an accepted candidate teaches the next pass."]
    for s in meta.get("streams") or []:
        span = ""
        if s.get("first"):
            span = (f" from {_when(s['first'])} to {_when(s['last'])} "
                    f"({(s['last'] - s['first']) / 60:.0f} min)")
        out.append(f"- Frames: {s['frames']}{span}, in `{s['dir']}`.")
    out += [f"- Every round trains from `{meta.get('base_detector')}` (the release weights); "
            f"the first teacher was `{meta.get('teacher')}`; the pipeline config's refiner is "
            "not trained.",
            *[f"- Resumed {r.get('at')}." for r in meta.get("resumes") or []],
            f"- Model config: `{p.get('model_config')}`. Lens: {meta.get('lens')} (CAMERA.K and "
            f"dist from `{meta.get('config')}`, copied into the session when it started).",
            "- Backgrounds for the synthetic share and the synthetic validation: "
            f"`{p.get('backgrounds')}` ({n_bgs} images).",
            "- The loop ends when a candidate is rejected (validation did not improve), when a "
            f"new teacher labels fewer than max(1, {p.get('min_growth', 0):.1%} of the previous "
            "pass) more frames (no more frames can be labelled), or after "
            f"{p.get('max_rounds')} rounds."]
    if loop and loop[-1].get("kind") == "exit":
        end = loop[-1]
        out.append(f"- Ended: {end['reason']}"
                   + (f" ({end['detail']})" if end.get("detail") else "")
                   + f"; best model `{end['best_onnx']}`. Nothing was deployed.")
    return out


def _short(path) -> str:
    return "/".join(Path(str(path)).parts[-4:])


def _loop_table(loop: list) -> list:
    """Tabulate each pass and its round: labels, growth, the gate and the ambiguity diagnostic.

    Args:
        loop: Lines of loop.jsonl.
    """
    rounds = {}
    for x in loop:
        if x.get("kind") == "round":
            rounds.setdefault(x["pass"], []).append(x)
    L = ["| pass | teacher | labelled train / held-out (anchors, tracked) | growth (needed) | "
         "relabel min | round | held-out recall teacher -> cand | far (<16 px) recall teacher "
         "-> cand | ambiguous poses base / teacher / cand | gate failed | decision |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ps in (x for x in loop if x.get("kind") == "pass"):
        growth = ("-" if ps.get("growth") is None
                  else f"{ps['growth']:+d} ({ps['growth_needed']})")
        head = (f"| {ps['pass']} | `{_short(ps['teacher'])}` | {ps['train']} / "
                f"{ps['heldout']} ({ps.get('anchor')}, {ps.get('tracked')}) | {growth} | "
                f"{ps.get('seconds', 0) / 60:.0f} | ")
        for r in rounds.get(ps["pass"]) or [None]:
            if r is None:
                L.append(head + f"- |||||{ps.get('decision', '')} |")
            elif "error" in r:
                L.append(head + f"{r['round']} | failed: {r['error']} |||| "
                         f"{r.get('decision', '')} |")
            else:
                rec, far = r["recall"], r["far_recall"]
                amb = r.get("ambiguous_rate") or {}
                failed = [k for k, ok in (r.get("gate") or {}).items() if not ok]
                L.append(head + f"{r['round']} | {_f(rec['teacher'])} -> {_f(rec['candidate'])} | "
                         f"{_f(far['teacher'])} -> {_f(far['candidate'])} | "
                         + " / ".join(_f(amb.get(k), "{:.1%}")
                                      for k in ("base", "teacher", "candidate"))
                         + f" | {', '.join(failed) or 'none'} | {r.get('decision', '')} |")
    return L


def record(session: Path, out: Path) -> Path:
    session, out = session.resolve(), out.resolve()
    root = session.parents[2]
    meta = json.loads((session / "session.json").read_text())
    resumes = meta.get("resumes") or []
    p = (resumes[-1].get("params") if resumes else None) or meta["params"]
    offline = meta.get("kind") == "offline"
    loop = _jsonl(session / "loop.jsonl")
    rounds = _jsonl(session / "rounds.jsonl")
    seen = {}
    for r in rounds:
        seen[r["round"]] = r
    rounds = [seen[k] for k in sorted(seen)]
    frames = _jsonl(session / "frames.jsonl")
    segs = [s for s in _jsonl(session / "segments.jsonl") if s.get("kind") == "segment"]
    seg_tot = {k: sum(s.get(k, 0) for s in segs)
               for k in ("labelled", "anchor", "tracked", "kept", "duplicate", "capped")}
    gens = generations(session)
    gen = gens[-1] if gens else 0
    banked = _banked(session, gen)
    relabelled = {}
    for r in _jsonl(session / "relabel.jsonl"):
        t = relabelled.setdefault(r.get("gen"), Counter())
        t.update({k: r.get(k, 0) for k in ("labelled", "anchor", "tracked", "kept")})
    n_raw = size_raw = 0
    for stream in ([_local(s["dir"], root) for s in meta.get("streams") or []] if offline
                   else [session / "stream"]):
        if stream.is_dir():
            with os.scandir(stream) as it:
                for e in it:
                    n_raw += 1
                    size_raw += e.stat().st_size
    ended = (session / "summary.json").exists()
    bgs = _local(p.get("backgrounds", ""), root)
    n_bgs = sum(1 for _ in bgs.iterdir()) if bgs.is_dir() else 0
    scored = [r for r in rounds if "reference" in r]
    last = scored[-1] if scored else (rounds[-1] if rounds else {})
    cfg_round = last.get("config") or {}
    ft = {}
    ft_path = session / "rounds" / f"round{last.get('round', 1)}" / "finetune.yaml"
    if ft_path.exists():
        import yaml
        ft = yaml.safe_load(ft_path.read_text()) or {}
    train = ft.get("train", {})
    rf = float(cfg_round.get("real_frac", p.get("real_frac")) or 0.0)
    finetune = ft.get("finetune", {})
    t0 = frames[0]["stamp"] if frames else None
    t1 = frames[-1]["stamp"] if frames else None
    ho = (_chunk_stats([_local(d, root) for d in scored[-1].get("heldout_dirs", [])]) if scored
          else banked["heldout"])
    ho_round = f"round {scored[-1]['round']}" if scored else "the bank so far"
    ho_anchor, ho_tracked = ho["sources"].get("anchor", 0), ho["sources"].get("tracked", 0)

    data, figs = out / "data", out / "figures"
    for d in (data, figs):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("session.json", "rounds.jsonl", "segments.jsonl", "summary.json", "loop.jsonl"):
        if (session / name).exists():
            shutil.copy2(session / name, data / name)
    if (session / "rounds" / "base_synth_val.json").exists():
        shutil.copy2(session / "rounds" / "base_synth_val.json", data / "base_synth_val.json")
    losses = {}
    for r in rounds:
        n = r["round"]
        rd = session / "rounds" / f"round{n}"
        for name in ("result.json", "synth_val.json", "finetune.yaml", "round.log"):
            if (rd / name).exists():
                shutil.copy2(rd / name, data / f"round{n}_{name}")
        m = _local(r.get("metrics", ""), root)
        if m.is_file():
            shutil.copy2(m, data / f"round{n}_metrics.jsonl")
            losses[n] = _losses(m)
    if (session / "dashboard.png").exists():
        shutil.copy2(session / "dashboard.png", figs / "dashboard.png")
    for pdf in sorted((session / "report").glob("*.pdf")) if (session / "report").is_dir() else []:
        shutil.copy2(pdf, figs / pdf.name)

    L = [f"# SSL fine-tune session {session.name}: record for evaluation", "",
         f"Generated {_when()} from `{session}` by "
         "`python3 -m convchart_ros.ssl_utils.record`; re-run it to refresh. The session had "
         + ("ended (summary.json written)." if ended else "not ended yet: the numbers are a snapshot."), "",
         "## What ran", "",
         "- Live self-supervised fine-tuning of the Conv-ChArT detector on the Pi camera stream "
         "(`task:=finetune`, `convchart_ros/ssl_node.py`).",
         f"- Frames from {_when(t0)} to {_when(t1)} ({(t1 - t0) / 60:.0f} min)." if t0 else "- No frames yet.",
         f"- Every round trains from `{meta.get('base_detector') or meta.get('detector')}` (the release weights); "
         f"the first teacher was `{meta.get('teacher') or meta.get('detector')}`; refiner `{meta.get('refiner')}`, "
         "not trained.",
         *[f"- Resumed {r.get('at')}: teacher `{r.get('teacher')}`, bank generation {r.get('generation')}, which "
           "relabels every stored frame with it." for r in resumes],
         f"- Model config: `{p.get('model_config')}`.",
         f"- Camera: the Pi's 640x480 mono PNG stream at 15 Hz; the teacher labels at most "
         f"{p.get('max_rate_hz')} Hz. Lens: {meta.get('lens')} (CAMERA.K and dist from `{meta.get('config')}`).",
         f"- Backgrounds for the synthetic replay and the synthetic validation: `{p.get('backgrounds')}` "
         f"({n_bgs} images).",
         f"- auto_deploy {p.get('auto_deploy')}: "
         + ("the teacher switches to each accepted round's model for the frames that follow; frames already banked "
            "keep their labels." if p.get("auto_deploy") else "the teacher stays the release model."), "",
         "## Settings", "",
         "| group | setting |", "|---|---|",
         ("| training | steps set by each round's bank: every banked frame drawn at least "
          f"{cfg_round.get('passes')} times ({cfg_round.get('sampling')} sampling), at least "
          f"{cfg_round.get('min_steps')} steps; " if cfg_round.get("passes") else
          f"| training | {train.get('steps')} steps; ")
         + f"batch {train.get('batch')} x accum {train.get('accum')}; lr peak "
         f"{train.get('lr')}, {train.get('warmup_steps')} warm-up steps, cosine to {train.get('lr_floor')}; wd "
         f"{train.get('wd')}; clip {train.get('clip_norm')}; EMA {train.get('ema_decay')}; freeze_bn "
         f"{finetune.get('freeze_bn')}; lr_mult {finetune.get('lr_mult')}; from the release weights every round |",
         f"| data mix | real_frac {rf}: {rf:.0%} of each batch is banked frames, the rest synthetic boards on the "
         "backgrounds, against forgetting; every round uses the whole train bank |",
         f"| harvest | anchors are frames whose identified corners fit the lattice; up to max_gap {p.get('max_gap')} "
         f"frames between anchors are labelled by tracking; near-duplicates (moved < {p.get('min_motion_px')} px) "
         f"skipped, at most {p.get('cap_per_bucket')} per bucket (0: every labelled frame kept); chunks of "
         f"{p.get('chunk_size')}; "
         + (f"held out: every {p.get('heldout_every')}th block of {p.get('split_block_s')} s of frame stamps, so a "
            "frame keeps its split whichever teacher labels it |" if p.get("split_block_s") else
            f"every {p.get('heldout_every')}th chunk held out |"),
         "| loop | each pass relabels every stored frame; a round follows each pass, on all "
         "its train frames; the loop stops at a rejected candidate, at growth under max(1, "
         f"{p.get('min_growth', 0):.1%}) of the previous pass, or after "
         f"{p.get('max_rounds')} rounds |" if offline else
         f"| round trigger | {p.get('min_train_frames')} train and {p.get('min_heldout_frames')} held-out frames, "
         f"then each time the train bank grows by {p.get('retrain_growth'):.0%} |",
         f"| acceptance | held-out recall up by >= {cfg_round.get('min_recall_gain')}, held-out ID precision down by "
         f"<= {cfg_round.get('max_precision_drop')}, synthetic M-04 down by <= {cfg_round.get('max_synth_m04_drop')}, "
         f"synthetic M-01 up by <= {cfg_round.get('max_synth_m01_rise')} px; synthetic validation on "
         f"{cfg_round.get('val_size')} images |", "",
         "## Data gathered", "",
         f"- Frames the teacher looked at: {len(frames)}; anchors {sum(bool(f.get('anchor')) for f in frames)} "
         f"({100 * sum(bool(f.get('anchor')) for f in frames) / max(1, len(frames)):.1f}%).",
         f"- Labelled: {seg_tot['labelled']} ({seg_tot['anchor']} anchors, {seg_tot['tracked']} by tracking); kept "
         f"{seg_tot['kept']}; dropped as near-duplicates {seg_tot['duplicate']}, over a bucket's cap "
         f"{seg_tot['capped']}.",
         f"- Banked: train {banked['train']['frames']} frames in {len(banked['train']['chunks'])} chunks "
         f"({banked['train']['sources']}); held-out {banked['heldout']['frames']} in "
         f"{len(banked['heldout']['chunks'])} ({banked['heldout']['sources']}).",
         f"- Raw frames saved (stream/): {n_raw}, {size_raw / 1e9:.2f} GB.",
         *[f"- Generation {g} (`{_bank_dir(session, g).relative_to(session)}`): banked train "
           f"{_banked(session, g)['train']['frames']}, held-out {_banked(session, g)['heldout']['frames']}"
           + (f"; its pass over the stored frames labelled {relabelled[g]['labelled']} "
              f"({relabelled[g]['anchor']} anchors, {relabelled[g]['tracked']} by tracking)" if g in relabelled else "")
           + "." for g in gens],
         f"- The tables below are generation {gen}'s bank.", "",
         "Banked frames by board square size and brightness (train / held-out), and how many were oblique "
         "(anisotropy > 0.15):", "",
         "| square | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for i, s in enumerate(SCALES):
        L.append(f"| {s} | " + " | ".join(f"{banked['train']['grid'][i, j]} / {banked['heldout']['grid'][i, j]}"
                                          for j in range(len(LEVELS))) + " |")
    L += ["", f"Oblique: train {banked['train']['oblique']}, held-out {banked['heldout']['oblique']}.", "",
          "## Rounds", "",
          "| round | started | min | train / held-out | steps (passes) | train loss first 50 -> last 50 steps | "
          "held-out recall ref -> cand | far (<16 px) recall ref -> cand | ID precision ref -> cand | fit rate ref "
          "-> cand | synth M-04 base -> cand | synth M-01 mean px base -> cand | "
          "ambiguous poses ref -> cand | gate failed | accepted |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        if "reference" not in r:
            L.append(f"| {r['round']} | {_when(r['t_start'])[11:16]} | {r.get('seconds', 0) / 60:.1f} | "
                     f"{r.get('train_frames')} / {r.get('heldout_frames')} | {r.get('steps', '-')} | failed: "
                     f"{r.get('error', '?')} ||||||||| no |")
            continue
        ref, cand = r["reference"]["all"], r["candidate"]["all"]
        far = [(x.get("far <16 px") or {}).get("recall") for x in (r["reference"], r["candidate"])]
        sb, sc = r.get("synth_base", {}), r.get("synth_candidate", {})
        lo = losses.get(r["round"], {})
        failed = [k for k, ok in (r.get("gate") or {}).items() if not ok]
        L.append(
            f"| {r['round']} | {_when(r['t_start'])[11:16]} | {r['seconds'] / 60:.1f} | {r['train_frames']} / "
            f"{r['heldout_frames']} | {r.get('steps', '-')} ({_f(r.get('passes'), '{:.1f}')}) | "
            f"{_f(lo.get('first50'), '{:.3f}')} -> {_f(lo.get('last50'), '{:.3f}')} | "
            f"{_f(ref['recall'])} -> {_f(cand['recall'])} | {_f(far[0])} -> {_f(far[1])} | "
            f"{_f(ref['id_precision'])} -> {_f(cand['id_precision'])} | "
            f"{_f(ref['fit_rate'], '{:.3f}')} -> {_f(cand['fit_rate'], '{:.3f}')} | "
            f"{_f((sb.get('m04') or {}).get('accuracy'))} -> {_f((sc.get('m04') or {}).get('accuracy'))} | "
            f"{_f((sb.get('m01') or {}).get('mean'), '{:.3f}')} -> {_f((sc.get('m01') or {}).get('mean'), '{:.3f}')}"
            f" | {_f(ref.get('ambiguous_rate'), '{:.1%}')} -> "
            f"{_f(cand.get('ambiguous_rate'), '{:.1%}')}"
            f" | {', '.join(failed) or 'none'} | {'yes' if r.get('accepted') else 'no'} |")
    if not rounds:
        L.append("| - | no round has finished yet ||||||||||||||")
    L += ["", "Reference = the model the candidate had to beat (the current teacher's model); base = the release "
          "weights. M-04 is synthetic ID accuracy; M-01 is the synthetic corner error in pixels. "
          "Ambiguous poses: the share of held-out frames with a pose whose two IPPE solutions "
          "src/inference.py could not tell apart (reprojection errors within 1.5x); a "
          "diagnostic, not part of the gate.", "",
          "## Caveats for evaluation", "",
          "1. **The held-out labels are pseudo-labels**, not hand labels. Of the "
          f"{ho['frames']} held-out frames {ho_round} was scored on, {ho_anchor} are the "
          + ("teacher's (the reference's)" if offline else "starting teacher's") + " own "
          f"detections (anchors) and "
          f"{ho_tracked} were labelled by tracking, where it could not fit the board. So the reference scores "
          "high nearly by construction; the fit-rate column shows how many of the tracked frames each model "
          "fits. An independent check needs hand-labelled frames, or frames from another session.",
          "2. **The held-out chunks come from the same session**, minutes apart from the training chunks, in the "
          "same room and light. That makes the held-out numbers optimistic.",
          f"3. **The held-out set is small**: {ho_round} was scored on {ho['frames']} frames, {ho['corners']} "
          f"labelled corners, so one "
          f"corner is {1 / max(1, ho['corners']):.4f} of recall and the acceptance margin is about "
          f"{round(cfg_round.get('min_recall_gain', 0) * ho['corners'])} corners. "
          + (f"Each pass holds out every {p.get('heldout_every')}th block of "
             f"{p.get('split_block_s')} s, as much of it as that pass's teacher labels."
             if offline else
             f"It only grows by a chunk every {p.get('heldout_every')} chunks banked (now "
             f"{banked['heldout']['frames']} frames), and later rounds are scored on the "
             "larger set."),
          "4. **Each round starts from the release weights** on "
          + ("all the train frames of its pass" if offline else "all frames banked so far")
          + "; rounds are not chained.",
          "5. **Diversity**: check the coverage table above. Buckets with no frames are views the fine-tune has "
          "not seen.",
          "6. The synthetic M-01 corner error is allowed to rise by up to the gate's margin per round; read it "
          "across rounds, not one at a time.", "",
          "## Files", "",
          "- `data/`: the session's own logs (session.json, rounds.jsonl, segments.jsonl, base_synth_val.json, "
          "summary.json once ended) and per round its result.json, synth_val.json, finetune.yaml, round.log and "
          "training metrics.jsonl.",
          "- `figures/`: the node's dashboard.png, and report/*.pdf once the session has ended.",
          f"- Not copied, in `{session}`: harvest/train and harvest/heldout (images.npy + records.json per chunk: "
          "the labelled frames), stream/ (every raw frame, named by stamp), rounds/round*/detector.onnx (each "
          "candidate). Checkpoints: Conv-ChArT/runs/ssl_<session>_r*/."]
    if offline:
        i = L.index("## What ran") + 2
        L[i:L.index("## Settings") - 1] = _offline_what(meta, p, loop, n_bgs)
        i = L.index("## Data gathered") + 2
        L[i:i + 4] = [f"- Stored frames: {n_raw} ({size_raw / 1e9:.2f} GB); every pass "
                      "relabels all of them.", L[i + 2]]
        i = L.index("## Rounds")
        L[i:i] = ["## Loop", "", *_loop_table(loop), ""]
        L.append("- Offline: `data/loop.jsonl` (a line per pass and per round, then the exit); "
                 "not copied: harvest/gen<g>/ (each pass's labelled frames; labels.jsonl names "
                 "each label's frame file).")
    (out / "README.md").write_text("\n".join(L) + "\n")
    return out / "README.md"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(record(a.session, a.out_dir))


if __name__ == "__main__":
    main()
