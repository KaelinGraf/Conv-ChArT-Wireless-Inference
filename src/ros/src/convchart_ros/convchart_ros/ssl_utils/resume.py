"""
Resuming an ssl fine-tune session (ROS-free): what a restarted node carries on from.

    session = latest_session(output_dir)      # the newest session folder, or None
    state = resume_state(session)             # its rounds, the model to beat and teach with, the next bank generation

A resumed node keeps the session folder: the frames saved in stream/, the banks in harvest/, the finished rounds in
rounds.jsonl (one per round number; a round logged twice keeps its last line) and the analytics clock (t0). The model
to beat, and the teacher, is the last accepted round's candidate, else the session's own starting teacher. Banks are
numbered generations, harvest/gen<g>/; a session from before generations has its bank at harvest/{train,heldout},
which counts as generation 0.
"""
from __future__ import annotations

import json
from pathlib import Path
import re


def latest_session(output_dir) -> Path | None:
    """Return the newest session folder (by name, which is its start time) under output_dir.

    Args:
        output_dir: Folder that holds the session folders.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    sessions = [p for p in root.iterdir() if p.is_dir() and (p / "session.json").exists()]
    return max(sessions, key=lambda p: p.name) if sessions else None


def read_rounds(session) -> list:
    """Return the session's finished rounds in round order, one per round number.

    Args:
        session: Session folder.
    """
    seen = {}
    path = Path(session) / "rounds.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and "round" in r:
                seen[r["round"]] = r
    return [seen[k] for k in sorted(seen)]


def generations(session) -> list:
    """Return the bank generations on disk, in order.

    Args:
        session: Session folder.
    """
    root = Path(session) / "harvest"
    if not root.is_dir():
        return []
    gens = sorted(int(m.group(1)) for d in root.iterdir() if d.is_dir() and (m := re.fullmatch(r"gen(\d+)", d.name)))
    if (root / "train").is_dir() or (root / "heldout").is_dir():
        gens = [0] + gens
    return gens


def resume_state(session) -> dict:
    """Return everything a node needs to carry on in session.

    The dict holds session, meta, history, round, best_onnx, teacher, next_gen and t0.

    Args:
        session: Session folder.
    """
    session = Path(session)
    meta = json.loads((session / "session.json").read_text())
    history = read_rounds(session)
    accepted = [r for r in history if r.get("accepted") and r.get("candidate_onnx")]
    best = accepted[-1]["candidate_onnx"] if accepted else None
    resumes = meta.get("resumes") or []
    started_with = (resumes[-1].get("teacher") if resumes else None) or meta.get("teacher") or meta.get("detector")
    gens = generations(session)
    analytics = session / "analytics.json"
    return {"session": str(session), "meta": meta, "history": history,
            "round": max((r["round"] for r in history), default=0), "best_onnx": best,
            "teacher": best or started_with, "next_gen": (max(gens) + 1) if gens else 1,
            "t0": json.loads(analytics.read_text()).get("t0") if analytics.exists() else None}
