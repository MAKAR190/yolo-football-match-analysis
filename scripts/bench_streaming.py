"""A/B benchmark for streaming optimizations. Not part of production code.

Two modes:

  1) Bench (default): run streaming pipeline once, report fps, optionally save
     per-frame ball_owner_id + final stats payload as a "baseline" pickle.

       python scripts/bench_streaming.py --every-n 1 --save-baseline bench_n1.pkl
       python scripts/bench_streaming.py --every-n 2 --save-baseline bench_n2.pkl

  2) Diff: compare two baseline pickles (frame-skip quality A/B). Reports the
     metrics from the streaming-speedup plan: ball_owner_id mismatch ratio,
     team possession deltas, ball visibility delta, pass attempts/completions
     delta.

       python scripts/bench_streaming.py --diff bench_n1.pkl bench_n2.pkl
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_bench(args):
    if args.no_ffmpeg_decode:
        # Monkey-patch so process_video_streaming's try-block raises and falls
        # back to cv2.VideoCapture. Cleanest way to A/B without code changes.
        import helpers
        def _broken(*a, **kw):
            raise RuntimeError("ffmpeg decode disabled by benchmark flag")
        helpers.iter_video_frames_ffmpeg_hwaccel = _broken
        import helpers.video as _hv
        _hv.iter_video_frames_ffmpeg_hwaccel = _broken

    from main import run_analysis_streaming

    timestamps = []
    owners = []
    def on_frame(annotated, idx, snap):
        timestamps.append(time.perf_counter())
        owners.append(int(snap.get("ball_owner_id", -1)))

    print(f"=== bench: half={args.half}, ffmpeg_decode={not args.no_ffmpeg_decode}, every_n={args.every_n}, threaded={not args.no_threading} ===")
    t0 = time.perf_counter()
    result = run_analysis_streaming(
        args.input,
        on_frame=on_frame,
        use_half=args.half,
        inference_every_n_frames=args.every_n,
        threaded=not args.no_threading,
    )
    elapsed = time.perf_counter() - t0
    n = result["frames_processed"]

    if len(timestamps) > args.warmup + 5:
        steady_t0 = timestamps[args.warmup]
        steady_n = len(timestamps) - args.warmup
        steady_dur = timestamps[-1] - steady_t0
        steady_fps = steady_n / steady_dur if steady_dur > 0 else 0.0
    else:
        steady_fps = float("nan")

    print(f"total: {n} frames in {elapsed:.2f}s = {n/elapsed:.1f} fps (incl. startup)")
    print(f"steady-state (after frame {args.warmup}): {steady_fps:.1f} fps")

    if args.save_baseline:
        # Load stats payload from the saved JSON so the pickle has everything
        # needed for a downstream diff.
        import json
        stats_payload = None
        stats_path = result.get("stats_path")
        if stats_path and Path(stats_path).exists():
            with open(stats_path, "r", encoding="utf-8") as fh:
                stats_payload = json.load(fh)
        baseline = {
            "input": args.input,
            "every_n": args.every_n,
            "use_half": args.half,
            "frames_processed": n,
            "elapsed_s": elapsed,
            "steady_fps": steady_fps,
            "ball_owner_per_frame": owners,
            "stats_payload": stats_payload,
        }
        out = Path(args.save_baseline)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(baseline, fh)
        print(f"saved baseline → {out}")


def _team_color_map(payload):
    """team_id_str -> (b, g, r) tuple."""
    if not payload:
        return {}
    raw = payload.get("team_colors_bgr") or {}
    return {str(k): tuple(int(c) for c in v) for k, v in raw.items()}


def _align_team_ids(colors_a, colors_b):
    """Return dict mapping team_id in B -> team_id in A by nearest color.

    K-means cluster ordering is nondeterministic across runs, so per-team
    stats can appear swapped even when the underlying values are stable.
    Aligning by color before diffing removes that artifact.
    """
    mapping = {}
    used = set()
    for tid_b, c_b in colors_b.items():
        best_tid = None
        best_d = float("inf")
        for tid_a, c_a in colors_a.items():
            if tid_a in used:
                continue
            d = sum((int(x) - int(y)) ** 2 for x, y in zip(c_b, c_a))
            if d < best_d:
                best_d = d
                best_tid = tid_a
        if best_tid is not None:
            mapping[tid_b] = best_tid
            used.add(best_tid)
    return mapping


def _team_possession_seconds(payload):
    if not payload:
        return {}
    poss = (payload.get("possession") or {}).get("team_possession_seconds") or {}
    return {str(k): float(v) for k, v in poss.items()}


def _remap(d, mapping):
    return {mapping.get(k, k): v for k, v in d.items()}


def _ball_visibility_ratio(payload):
    if not payload:
        return None
    vis = payload.get("ball_visibility") or {}
    if "ball_visibility_ratio" in vis:
        return float(vis["ball_visibility_ratio"])
    return None


def _pass_counts(payload):
    if not payload:
        return {}
    src = payload.get("pass_entry") or {}
    attempts = {str(k): int(v) for k, v in (src.get("team_pass_attempts") or {}).items()}
    completed = {str(k): int(v) for k, v in (src.get("team_pass_completed") or {}).items()}
    teams = set(attempts) | set(completed)
    return {t: {"attempts": attempts.get(t, 0), "completed": completed.get(t, 0)} for t in teams}


def run_diff(args):
    a_path, b_path = args.diff
    with open(a_path, "rb") as fh:
        a = pickle.load(fh)
    with open(b_path, "rb") as fh:
        b = pickle.load(fh)

    print(f"=== diff: A={Path(a_path).name} (every_n={a.get('every_n')}) vs B={Path(b_path).name} (every_n={b.get('every_n')}) ===")
    print(f"fps:  A={a.get('steady_fps', float('nan')):.1f}  B={b.get('steady_fps', float('nan')):.1f}  speedup={b.get('steady_fps', 0)/max(a.get('steady_fps', 1e-9), 1e-9):.2f}x")

    owners_a = a.get("ball_owner_per_frame") or []
    owners_b = b.get("ball_owner_per_frame") or []
    n = min(len(owners_a), len(owners_b))
    if n == 0:
        print("no owner data in baselines")
    else:
        diff_count = sum(1 for i in range(n) if owners_a[i] != owners_b[i])
        print(f"ball_owner_id mismatch: {diff_count}/{n} frames = {100.0*diff_count/n:.2f}%  (target <= 3%)")

    colors_a = _team_color_map(a.get("stats_payload"))
    colors_b = _team_color_map(b.get("stats_payload"))
    b_to_a = _align_team_ids(colors_a, colors_b) if colors_a and colors_b else {}
    if b_to_a:
        print(f"team alignment (B->A by nearest color): {b_to_a}")

    pa = _team_possession_seconds(a.get("stats_payload"))
    pb_raw = _team_possession_seconds(b.get("stats_payload"))
    pb = _remap(pb_raw, b_to_a)
    if pa and pb:
        for team in sorted(set(pa) | set(pb)):
            va, vb = pa.get(team, 0.0), pb.get(team, 0.0)
            denom = max(va, 1e-9)
            pct = 100.0 * (vb - va) / denom
            print(f"possession[{team}]: A={va:.2f}s  B={vb:.2f}s  delta={pct:+.2f}%  (target +/-2%)")
    else:
        print("possession: not available in baseline payloads")

    va = _ball_visibility_ratio(a.get("stats_payload"))
    vb = _ball_visibility_ratio(b.get("stats_payload"))
    if va is not None and vb is not None:
        delta = 100.0 * (vb - va) / max(va, 1e-9)
        print(f"ball_visibility_ratio: A={va:.4f}  B={vb:.4f}  delta={delta:+.2f}%  (target +/-1%)")
    else:
        print("ball_visibility_ratio: not available in baseline payloads")

    ca = _pass_counts(a.get("stats_payload"))
    cb_raw = _pass_counts(b.get("stats_payload"))
    cb = _remap(cb_raw, b_to_a)
    if ca and cb:
        for team in sorted(set(ca) | set(cb)):
            xa = ca.get(team, {"attempts": 0, "completed": 0})
            xb = cb.get(team, {"attempts": 0, "completed": 0})
            d_att = 100.0 * (xb["attempts"] - xa["attempts"]) / max(xa["attempts"], 1)
            d_comp = 100.0 * (xb["completed"] - xa["completed"]) / max(xa["completed"], 1)
            print(f"passes[{team}]: A={xa}  B={xb}  deltaattempts={d_att:+.2f}%  deltacompleted={d_comp:+.2f}%  (target +/-5%)")
    else:
        print("passes: not available in baseline payloads")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs/Video Project 1.mp4")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--no-ffmpeg-decode", action="store_true", help="Force cv2.VideoCapture decode")
    parser.add_argument("--warmup", type=int, default=15, help="Frames to skip when computing steady-state fps")
    parser.add_argument("--every-n", type=int, default=1, help="Run YOLO+ByteTrack every Nth frame; reuse on others")
    parser.add_argument("--no-threading", action="store_true", help="Disable producer/consumer threading (sequential loop)")
    parser.add_argument("--save-baseline", default=None, help="Pickle path to save per-frame owner + stats for later --diff")
    parser.add_argument("--diff", nargs=2, metavar=("BASELINE_A", "BASELINE_B"), default=None,
                        help="Compare two baseline pickles instead of running the pipeline")
    args = parser.parse_args()

    if args.diff:
        run_diff(args)
    else:
        run_bench(args)


if __name__ == "__main__":
    main()
