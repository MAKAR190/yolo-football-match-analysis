"""Smoke test for run_analysis_streaming. Not part of production code."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_analysis_streaming

calls = []


def on_frame(annotated, idx, snap):
    calls.append(idx)
    if len(calls) <= 5 or len(calls) % 30 == 0:
        print(
            f"  frame {idx}: calib={snap.get('calibrating')}, "
            f"n_players={snap.get('n_players')}, "
            f"owner={snap.get('ball_owner_id')}, "
            f"shape={annotated.shape}"
        )


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/Video Project.mp4"
    t0 = time.perf_counter()
    result = run_analysis_streaming(input_path, on_frame=on_frame)
    elapsed = time.perf_counter() - t0
    n = result["frames_processed"]
    print(f"\nfinished {n} frames in {elapsed:.1f}s ({n/max(elapsed, 1e-6):.1f} fps)")
    print(f"on_frame called {len(calls)} times")
    print(f"output: {result['output_video_path']}")
    print(f"stats:  {result['stats_path']}")


if __name__ == "__main__":
    main()
