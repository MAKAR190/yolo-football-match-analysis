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

## How to run

**Prerequisites:** Python 3 with GPU recommended for real-time-ish runs. Typical dependencies include `ultralytics`, `torch`, `supervision`, `opencv-python`, `numpy`, `scikit-learn`, and `streamlit` for the UI. Install what your environment needs (this repo does not ship a single pinned lockfile).

**Batch / CLI — process the default demo video:**

```bash
python main.py
```

This uses `inputs/football_input.mp4` (see `main.py`) and writes e.g. `outputs/<stem>_output.mp4`. To analyse another file, import and call `run_analysis("path/to/video.mp4")` or adjust the script.

**Streamlit — browse inputs and stats:**

```bash
streamlit run streamlit_stats.py
```

## Limitations (inherent to the goal)

Broadcast video is **not** calibrated: distances in pixels are not metres on the pitch. Occlusion, motion blur, and small ball size limit ball and touch accuracy. Stats here are **experimental** and best read as feasibility checks for “can we get *something* useful from one camera?” rather than as official match data.
