"""Live analytics for the SSL node: JSONL logs, a rolling summary and dashboard figures (ROS-free).

    a = Analytics(session_dir)                      # resume=True: keep a session's clock and history, append
    a.frame(event)                                  # each harvester event (+ latency_ms, arrival); to frames.jsonl
    a.segment(seg, seq, kind)                       # a segment outside a frame event (the flush at shutdown)
    a.relabel(seg, gen, seq)                        # a segment of a bank generation's relabel pass; relabel.jsonl
    a.round_done(result)                            # each finished round; appended to rounds.jsonl
    a.summary()                                     # rolling stats over window_s seconds, for the ssl/stats topic
    a.render(coverage, rounds, metrics_paths)       # dashboard.png bytes; pdf=True also writes report/*.pdf

    python3 -m convchart_ros.ssl_utils.analytics <session dir>   # re-render dashboard.png and report/ from the logs

Panels, with training rounds shaded on the time axes: stream rate and teacher latency; cumulative frames, anchors,
labelled and kept frames; kept-frame coverage by scale and brightness; label checks (closure and forward-backward
errors, alignment correlation); training loss per round; held-out recall and ID precision per round against the model
it had to beat.
"""
from __future__ import annotations

import argparse
from collections import deque
import io
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

SCALES = ["<16", "16-32", "32-64", ">=64"]
LEVELS = ["dark", "dim", "bright"]


def _read_jsonl(path: Path) -> list:
    """The rows of a JSONL log; a line that does not parse (cut short by a crash) is skipped."""
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class Analytics:

    def __init__(self, session_dir, window_s: float = 30.0, replay: bool = False, resume: bool = False):
        """
        replay: re-render a finished session from its logs, writing nothing but the figures. resume: carry on a
        session, keeping its clock (t0) and history and appending to its logs.
        """
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.window_s = window_s
        self.lock = threading.Lock()
        self.frames: deque = deque(maxlen=500_000)
        self.segments: deque = deque(maxlen=100_000)
        self.rounds: list = []
        self.replay = replay
        meta = self.dir / "analytics.json"
        if replay or (resume and meta.exists()):
            self.t0 = (json.loads(meta.read_text())["t0"] if meta.exists()
                       else (self.dir / "session.json").stat().st_mtime)
            rows = lambda name: _read_jsonl(self.dir / name)
            self.frames.extend((r["t"], r.get("latency_ms", np.nan), bool(r["anchor"]), r.get("n_ids", 0))
                               for r in rows("frames.jsonl"))
            self.segments.extend(rows("segments.jsonl"))
            done = {r["round"]: r for r in rows("rounds.jsonl") if "round" in r}
            self.rounds = [done[k] for k in sorted(done)]
            if replay:
                self._files = ()
                return
        else:
            self.t0 = time.time()
            meta.write_text(json.dumps({"t0": self.t0, "window_s": window_s}))
        self._frames_f = open(self.dir / "frames.jsonl", "a", buffering=1)
        self._segments_f = open(self.dir / "segments.jsonl", "a", buffering=1)
        self._rounds_f = open(self.dir / "rounds.jsonl", "a", buffering=1)
        self._relabel_f = open(self.dir / "relabel.jsonl", "a", buffering=1)
        self._files = (self._frames_f, self._segments_f, self._rounds_f, self._relabel_f)

    def frame(self, ev: dict) -> None:
        now = time.time() - self.t0
        t = ev["arrival"] - self.t0 if "arrival" in ev else now
        row = {k: v for k, v in ev.items() if k not in ("segment", "break", "arrival")}
        row["t"], row["lag_s"] = round(t, 3), round(now - t, 3)
        if "break" in ev:
            row["stream_break"] = True
        with self.lock:
            self.frames.append((t, ev.get("latency_ms", np.nan), bool(ev["anchor"]), ev.get("n_ids", 0)))
            self._frames_f.write(json.dumps(row) + "\n")
        for kind in ("break", "segment"):
            if ev.get(kind):
                self.segment(ev[kind], ev["seq"], kind, now)

    def segment(self, seg: dict, seq: int, kind: str = "segment", now: float | None = None) -> None:
        now = time.time() - self.t0 if now is None else now
        seg = {**seg, "t": round(now, 3), "seq": seq, "kind": kind}
        with self.lock:
            self.segments.append(seg)
            self._segments_f.write(json.dumps(seg) + "\n")

    def relabel(self, seg: dict, gen: int, seq: int) -> None:
        """A segment of a bank generation's pass over the stored frames: to relabel.jsonl, kept out of the live
        stream's tallies (those frames were counted when they came)."""
        row = {**seg, "t": round(time.time() - self.t0, 3), "seq": seq, "gen": gen, "kind": "relabel"}
        with self.lock:
            self._relabel_f.write(json.dumps(row) + "\n")

    def round_done(self, result: dict) -> None:
        with self.lock:
            if any(r.get("round") == result.get("round") for r in self.rounds):
                return
            self.rounds.append(result)
            self._rounds_f.write(json.dumps(result, default=str) + "\n")

    def summary(self) -> dict:
        now = time.time() - self.t0
        with self.lock:
            fr = [f for f in self.frames if f[0] >= now - self.window_s]
            sg = [s for s in self.segments if s["t"] >= now - self.window_s]
            n_all = len(self.frames)
            labelled = sum(s["labelled"] for s in self.segments)
            kept = sum(s["kept"] for s in self.segments)
        lat = np.array([f[1] for f in fr], dtype=float)
        lat = lat[np.isfinite(lat)]
        span = (fr[-1][0] - fr[0][0]) if len(fr) > 1 else 0.0
        return {"uptime_s": round(now, 1), "frames": n_all, "labelled": labelled, "kept": kept,
                "fps": round((len(fr) - 1) / span, 2) if span > 0 else 0.0,
                "latency_ms_p50": round(float(np.median(lat)), 1) if lat.size else None,
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat.size else None,
                "anchor_rate": round(float(np.mean([f[2] for f in fr])), 3) if fr else None,
                "labelled_per_s": round(sum(s["labelled"] for s in sg) / self.window_s, 2),
                "kept_per_s": round(sum(s["kept"] for s in sg) / self.window_s, 2)}

    def close(self) -> None:
        with self.lock:
            for f in self._files:
                f.close()

    def render(self, coverage: dict, rounds: dict | None = None, metrics_paths: dict | None = None,
               pdf: bool = False) -> bytes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with self.lock:
            fr = np.array([(f[0], f[1], f[2]) for f in self.frames], dtype=float).reshape(-1, 3)
            sg = list(self.segments)
            done = list(self.rounds)
        now = time.time() - self.t0
        if self.replay:
            now = max(list(fr[-1:, 0]) + [s["t"] for s in sg[-1:]]
                      + [r["t_end"] - self.t0 for r in done if "t_end" in r] + [0.0])
        spans = [(r["t_start"] - self.t0, r["t_end"] - self.t0) for r in done if "t_start" in r]
        if rounds and rounds.get("phase") not in (None, "idle") and rounds.get("started"):
            spans.append((rounds["started"] - self.t0, now))
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
        fig.suptitle(f"Conv-ChArT live self-supervision   session {self.dir.name}   t = {now:.0f} s   "
                     f"frames {len(fr)}   labelled {sum(s['labelled'] for s in sg)}   "
                     f"kept {sum(s['kept'] for s in sg)}", fontsize=12)
        extra = {(0, 0): self._p_stream(axes[0, 0], fr, now)}
        self._p_counts(axes[0, 1], fr, sg, now)
        for ax in (axes[0, 0], axes[0, 1]):
            for i, (a, b) in enumerate(spans):
                ax.axvspan(a, b, color="tab:gray", alpha=0.15, lw=0, label="training round" if i == 0 else None)
        if spans:
            axes[0, 0].legend(loc="lower left", fontsize=8)
        if len(fr) or spans:
            axes[0, 1].legend(loc="upper left", fontsize=8)
        extra[(0, 2)] = self._p_coverage(fig, axes[0, 2], coverage)
        extra[(1, 0)] = self._p_checks(axes[1, 0], sg)
        self._p_training(axes[1, 1], rounds, metrics_paths or {})
        self._p_rounds(axes[1, 2], done)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        png = buf.getvalue()
        tmp = self.dir / "dashboard.png.tmp"
        tmp.write_bytes(png)
        tmp.replace(self.dir / "dashboard.png")
        if pdf:
            rep = self.dir / "report"
            rep.mkdir(exist_ok=True)
            fig.savefig(rep / "dashboard.pdf")
            groups = [[axes[i, j], *extra.get((i, j), [])] for i in range(2) for j in range(3)]
            self._save_panels(fig, groups, rep)
        plt.close(fig)
        return png

    @staticmethod
    def _save_panels(fig, groups, out_dir, dpi: int = 150, pad: float = 0.1) -> list:
        import re
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        state = [(o, o.get_visible()) for o in [*fig.axes, *fig.texts]]
        bbs = [Bbox.union([b for b in (a.get_tightbbox(r) for a in group) if b is not None])
               .transformed(fig.dpi_scale_trans.inverted()) for group in groups]
        paths = []
        for i, (group, bb) in enumerate(zip(groups, bbs), 1):
            for o, _ in state:
                o.set_visible(any(o is a for a in group))
            slug = re.sub(r"[^a-z0-9]+", "_", group[0].get_title().lower()).strip("_")[:40].rstrip("_")
            paths.append(Path(out_dir) / f"panel_{i}_{slug}.pdf")
            fig.savefig(paths[-1], dpi=dpi, bbox_inches=bb.padded(pad))
        for o, v in state:
            o.set_visible(v)
        return paths

    @staticmethod
    def _bin(t, v, width):
        """The mean of v in bins of width over t, from t's first sample (NaN in a bin with no finite v)."""
        if len(t) == 0:
            return np.array([]), np.array([])
        lo = float(np.min(t))
        idx = np.floor((np.asarray(t) - lo) / width).astype(int)
        n = int(idx.max()) + 1
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v)
        sums = np.bincount(idx[ok], weights=v[ok], minlength=n)
        counts = np.bincount(idx[ok], minlength=n)
        vals = np.full(n, np.nan)
        vals[counts > 0] = sums[counts > 0] / counts[counts > 0]
        return lo + (np.arange(n) + 0.5) * width, vals

    def _p_stream(self, ax, fr, t_end):
        ax.set_title("Stream: processed rate and teacher latency")
        if len(fr) > 1:
            t_end = max(t_end, fr[-1, 0])
            width = max(2.0, (t_end - fr[0, 0]) / 120)
            edges = np.arange(fr[0, 0], t_end + width, width)
            counts, _ = np.histogram(fr[:, 0], edges)
            ax.plot(edges[:-1] + width / 2, counts / width, color="tab:blue", label="frames/s")
            ax2 = ax.twinx()
            c, lat = self._bin(fr[:, 0], fr[:, 1], width)
            ax2.plot(c, lat, color="tab:orange", label="latency ms")
            ax2.set_ylabel("teacher latency (ms)", color="tab:orange")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frames per second", color="tab:blue")
        return [ax2] if len(fr) > 1 else []

    def _p_counts(self, ax, fr, sg, t_end):
        ax.set_title("Labelling: cumulative frames")
        if len(fr):
            t_end = max(t_end, fr[-1, 0])
            t = np.append(fr[:, 0], t_end)
            ax.step(t, np.append(np.arange(1, len(fr) + 1), len(fr)), where="post", color="tab:blue",
                    label="processed")
            ax.step(t, np.append(np.cumsum(fr[:, 2]), fr[:, 2].sum()), where="post", color="tab:green",
                    label="anchors")
        if sg:
            st = np.append([s["t"] for s in sg], t_end)
            for key, col in (("labelled", "tab:purple"), ("kept", "tab:red")):
                n = np.cumsum([s[key] for s in sg])
                ax.step(st, np.append(n, n[-1]), where="post", color=col, label=key)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frames")

    def _p_coverage(self, fig, ax, coverage):
        ax.set_title("Kept frames by scale and brightness")
        grid = np.zeros((len(SCALES), len(LEVELS)))
        for key, n in (coverage or {}).items():
            s, b, _ = (int(x) for x in key.split(","))
            grid[s, b] += n
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(len(SCALES)):
            for j in range(len(LEVELS)):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center",
                        color="white" if grid[i, j] < grid.max() / 2 else "black")
        ax.set_xticks(range(len(LEVELS)), LEVELS)
        ax.set_yticks(range(len(SCALES)), SCALES)
        ax.set_xlabel("board brightness")
        ax.set_ylabel("square size (px)")
        return [fig.colorbar(im, ax=ax, fraction=0.046).ax]

    def _p_checks(self, ax, sg):
        ax.set_title("Label checks: tracking errors and alignment")
        clos = [e for s in sg for e in s["closures"]]
        fb = [e for s in sg for e in s["fb"]]
        rho = [r for s in sg for r in s["rho"]]
        bins = np.linspace(0, 1.5, 31)
        if clos:
            ax.hist(np.clip(clos, 0, 1.5), bins=bins, alpha=0.6, label=f"closure (n={len(clos)})")
        if fb:
            ax.hist(np.clip(fb, 0, 1.5), bins=bins, alpha=0.6, label=f"forward-backward (n={len(fb)})")
        ax.set_xlabel("disagreement (px)")
        ax.set_ylabel("tracks")
        extra = []
        if rho:
            ax2 = ax.twinx()
            ax2.hist(rho, bins=np.linspace(0, 1, 41), histtype="step", color="k", label="ECC rho")
            ax2.set_ylabel("labelled frames (ECC rho, line)")
            extra.append(ax2)
        if clos or fb:
            ax.legend(loc="upper right", fontsize=8)
        return extra

    def _p_training(self, ax, rounds, metrics_paths):
        ax.set_title("Training loss per round")
        for name, path in sorted(metrics_paths.items()):
            p = Path(path)
            if not p.exists():
                continue
            steps, loss = [], []
            for line in p.read_text().splitlines():
                rec = json.loads(line) if line.strip() else {}
                if "loss" in rec and "step" in rec:
                    steps.append(rec["step"])
                    loss.append(rec["loss"])
            if steps:
                k = max(1, len(loss) // 60)
                sm = np.convolve(loss, np.ones(k) / k, mode="valid")
                ax.plot(steps[k - 1:], sm, label=name)
        if rounds and rounds.get("phase") not in (None, "idle"):
            ax.text(0.02, 0.95, f"round {rounds['round']}: {rounds['phase']} {100 * rounds['progress']:.0f}%",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        if ax.lines:
            ax.legend(fontsize=8)

    def _p_rounds(self, ax, done):
        ax.set_title("Held-out recall / ID precision per round")
        ok = [r for r in done if "candidate" in r]
        if ok:
            x = np.arange(len(ok))
            ref = [r["reference"].get("all", {}).get("recall") or 0 for r in ok]
            cand = [r["candidate"].get("all", {}).get("recall") or 0 for r in ok]
            dark_r = [(r["reference"].get("dark") or {}).get("recall") for r in ok]
            dark_c = [(r["candidate"].get("dark") or {}).get("recall") for r in ok]
            prec = [r["candidate"].get("all", {}).get("id_precision") or 0 for r in ok]
            ax.bar(x - 0.2, ref, 0.4, label="model to beat", color="tab:gray")
            ax.bar(x + 0.2, cand, 0.4, label="candidate", color=["tab:green" if r["accepted"] else "tab:red"
                                                                for r in ok])
            ax.plot(x, prec, "k^", label="candidate ID precision")
            if any(v is not None for v in dark_c):
                ax.plot(x - 0.2, [v if v is not None else np.nan for v in dark_r], "o", color="tab:gray", mfc="none",
                        label="dark recall (to beat)")
                ax.plot(x + 0.2, [v if v is not None else np.nan for v in dark_c], "o", color="tab:blue", mfc="none",
                        label="dark recall (candidate)")
            ax.set_xticks(x, [f"r{r['round']}" + (" ok" if r["accepted"] else "") for r in ok])
            ax.set_ylim(0, 1.32)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.legend(fontsize=7, loc="upper center", ncol=3)
        failed = [r for r in done if "error" in r]
        if failed:
            ax.text(0.02, 0.02, f"{len(failed)} round(s) failed: {failed[-1]['error'][:60]}", transform=ax.transAxes,
                    fontsize=7, color="tab:red")
        ax.set_ylabel("held-out (pseudo-label) rate")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    session = Path(a.session)
    info = json.loads((session / "session.json").read_text())
    if info.get("convchart_root") and info["convchart_root"] not in sys.path:
        sys.path.insert(0, info["convchart_root"])
    summary = session / "summary.json"
    coverage = json.loads(summary.read_text()).get("coverage") if summary.exists() else None
    if coverage is None:
        from dcc.video import bucket_of
        coverage = {}
        for f in sorted(session.glob("harvest/*/chunk_*/records.json")):
            for rec in json.loads(f.read_text())["records"]:
                key = ",".join(map(str, bucket_of(rec)))
                coverage[key] = coverage.get(key, 0) + 1
    an = Analytics(session, replay=True)
    metrics = {f"r{r['round']}": r["metrics"] for r in an.rounds if r.get("metrics")}
    an.render(coverage, None, metrics, pdf=not a.no_pdf)
    print(f"rendered {session / 'dashboard.png'}" + ("" if a.no_pdf else f" and {session / 'report'}"))


if __name__ == "__main__":
    main()
