# YOLO football match analysis

## Goal

This project explores **whether simple football match statistics can be estimated from a single broadcast-style camera feed**—the kind of wide, panning TV shot you get from one fixed sideline or elevated camera—**without** dedicated multi-camera tracking or pitch instrumentation. The idea is to see how far object detection, tracking, and heuristics can go when the only input is ordinary match video.

What “simple stats” means here is things derivable from **who has the ball**, **team colours**, and **geometry in image space**: for example ball possession time, rough touch or pass cues, and visibility-limited ball statistics, rather than full tactical event data from calibrated pitch coordinates.

## What the repo does

1. **Detection** — A custom Ultralytics YOLO model (`models/`, e.g. `best_soccanna.pt`) finds players, referees, and the ball in each frame.
2. **Tracking** — [Supervision](https://github.com/roboflow/supervision) **ByteTrack** assigns stable IDs across frames so the same player or ball instance can be followed over time without loading the whole video into RAM (batched inference + streaming frames).
3. **Teams** — Jersey colours are clustered (e.g. K-means) so detections can be split into two outfield teams for possession-style stats.
4. **Ball ↔ player logic** — Heuristics link the ball to the most plausible player (distance, bbox overlap, motion cues), with options for hybrid ball tracking when raw detections and tracker disagree.
5. **Broadcast camera** — A **camera movement** estimator can compensate for pan/zoom so stats are less wrong when the picture moves.
6. **Outputs** — An annotated video under `outputs/` and JSON stats (e.g. under `outputs/stats/`) for downstream viewing or experiments.

Core orchestration lives in `modules/trackers.py` (`ByteTracker`); high-level entry and default parameters are in `main.py`.

## Repository layout (short)

| Path | Role |
|------|------|
| `main.py` | `run_analysis()` — runs the pipeline; CLI defaults to a demo input under `inputs/`. |
| `modules/` | Tracker, team assigner, ball–player assigner, camera movement, stats analyses. |
| `helpers/` | Video I/O, geometry, foot/ball ROI helpers. |
| `inputs/` | Place input videos here for the default workflow. |
| `outputs/` | Rendered videos and generated stats. |
| `models/` | YOLO weights. |
| `streamlit_stats.py` | Optional UI to pick videos and inspect stats. |

## Installation

### Prerequisites

- **Python 3.12** (developed on 3.12.8; 3.10+ should work).
- A **CUDA-capable GPU is strongly recommended.** The pipeline runs on CPU, but at a fraction of the throughput — `main.py` ships with GPU defaults (`device="cuda:0"`). On CPU, lower the inference settings (see [Tuning for CPU](#tuning-for-cpu)).
- **ffmpeg** is optional. The hardware-accelerated encode/decode paths use it, but the repo falls back to the binary bundled with `imageio-ffmpeg` if no system ffmpeg is on `PATH`.
- **YOLO weights.** The default is `models/best_soccanna.pt` (custom-trained on a Roboflow football dataset). It must be present in `models/`.

### Set up the environment

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd yolo-football-match-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install ultralytics supervision opencv-python torch numpy scikit-learn streamlit imageio-ffmpeg
```

> **GPU note:** to run on a GPU, install the CUDA build of PyTorch that matches your CUDA toolkit *before* (or instead of) the plain `torch` above — see the selector at <https://pytorch.org/get-started/locally/>. With a CPU-only `torch`, change `device="cuda:0"` to `device="cpu"` in `main.py`.

This repo intentionally ships **no pinned lockfile** — install the versions your environment needs.

## Usage

### CLI — process a video

`run_analysis()` in [main.py](main.py) is the canonical entry point. It writes an annotated video to `outputs/<stem>_output.mp4` and a stats JSON to `outputs/stats/<stem>_output.stats.json`.

The bare `python main.py` expects a file at `inputs/football_input.mp4`. To analyse one of your own clips, call `run_analysis` with its path:

```bash
python -c "from main import run_analysis; run_analysis('inputs/Video Project.mp4')"
```

Or, optionally, choose a different model / output location:

```python
from main import run_analysis

run_analysis(
    input_path="inputs/Video Project.mp4",
    model_path="models/best_soccanna.pt",   # default
    output_video_path="outputs/my_run.mp4",  # default: outputs/<stem>_output.mp4
)
```

The first run on a clip is the slowest (it runs the full track pass); subsequent renders reuse cached tracks from `cache/` when caching is enabled.

### Streamlit — browse inputs and inspect stats

```bash
streamlit run streamlit_stats.py
```

This opens a UI that lists processed videos and renders the generated stats (possession, touches, pass cues, ball visibility) from `outputs/stats/`.

### Outputs

| Artifact | Location |
|----------|----------|
| Annotated video | `outputs/<stem>_output.mp4` |
| Stats JSON | `outputs/stats/<stem>_output.stats.json` |
| Cached tracking results | `cache/` (safe to delete to force re-tracking) |

### Tuning for CPU

GPU defaults live in `run_analysis` / the config dataclasses in [modules/trackers.py](modules/trackers.py). When running CPU-first, start conservative: `scale=0.5`, `step=2`, `batch=30`, and `device="cpu"`. Increase resolution/batch only once a short clip runs end to end.

## Limitations (inherent to the goal)

Broadcast video is **not** calibrated: distances in pixels are not metres on the pitch. Occlusion, motion blur, and small ball size limit ball and touch accuracy. Stats here are **experimental** and best read as feasibility checks for “can we get *something* useful from one camera?” rather than as official match data.
