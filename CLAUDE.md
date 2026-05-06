# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Estimate simple football match statistics (possession, touches, pass cues, ball visibility) from a **single broadcast-style camera feed**, using YOLO detection + ByteTrack + heuristics. Stats are derived from "who has the ball", team colours, and image-space geometry — there is no pitch calibration, so pixel distances are not metres.

## Commands

- Run the full pipeline on the default demo video (`inputs/football_input.mp4`):
  ```bash
  python main.py
  ```
  Writes annotated video to `outputs/<stem>_output.mp4` and stats JSON to `outputs/stats/<stem>_output.stats.json`.
- Run on a different file: `from main import run_analysis; run_analysis("path/to/video.mp4")`. `run_analysis` is the canonical entry point used by both CLI and Streamlit.
- Streamlit UI to browse inputs and stats:
  ```bash
  streamlit run streamlit_stats.py
  ```
- There is no test suite, no lockfile, and no linter configured. `models/best_soccanna.pt` is the default YOLO weights file (custom-trained on a Roboflow football dataset). Class IDs are `player=0, ball=1, referee=2` (see `modules/trackers.py`).

## Hard architectural constraint: stream, never buffer

From `.specify/memory/constitution.md` — this is non-negotiable:

- **Never** load all frames into memory (`frames.append(frame)` until EOF is forbidden).
- Use `helpers.video.iter_video_frames` / `iter_video_frames_adaptive` / `iter_video_frames_ffmpeg_hwaccel` and bounded YOLO inference batches (e.g. `batch=30`).
- If a computation needs the whole clip (e.g. team-colour clustering needs collected crops), do **two streaming passes**: pass 1 streams to build small artifacts and `tracks`; pass 2 streams again to render annotations using `tracks`. This is exactly the pattern `ByteTracker.process_video` implements (`get_tracks_from_video` then `render_video_from_tracks`).
- Cache key correctness: any tracking cache must include model paths, video identity, and all preprocessing/inference params. See `ByteTracker._cache_key_path` and bump `ByteTracker.PIPELINE_VERSION` when detection/postprocessing logic changes so old caches are invalidated.

## Pipeline shape

`main.run_analysis` → `ByteTracker(model).process_video(...)` runs two streaming passes:

1. **Track pass** (`get_tracks_from_video`): YOLO predicts in batches → `supervision.ByteTrack` assigns stable IDs → per-frame dicts of player/referee/ball tracks are accumulated. Ball detections go through a prefilter (`ball_candidate_prefilter`) and a temporal outlier guard (`BallOutputOutlierRejection`) so a glare/foot blob can't latch as the ball. `ball_tracking_mode` ∈ {`raw_candidates`, `byte_track`, `hybrid`} controls whether ball comes from raw YOLO dets, ByteTrack output, or hybrid (default in `main.py`).
2. **Render pass** (`render_video_from_tracks`): re-streams the source video frame by frame; for each frame it runs `TeamAssigner` (jersey K-means) → `PlayerBallAssigner` (ball→player) → `CameraMovementEstimator` (LK optical flow) → `Annotations` overlays → `StatsManager.update`. Output is written through `cv2.VideoWriter` or an ffmpeg hardware encoder (`hw_encoder="h264_mf"` etc.).

`RenderOwnershipStatsConfig` (frozen dataclass at the top of `modules/trackers.py`) is the single source of truth for ownership/possession/touch heuristics — `main.py` instantiates it with the tuned defaults. Don't add ownership knobs elsewhere; add them here.

## Module map (only the non-obvious stuff)

- `modules/trackers.py` (~2k lines) is the orchestrator. `ByteTracker` owns the tracker config, both passes, ball prefiltering, the rendered-ball outlier guard, and the ffmpeg-based GPU render pipeline. Most pipeline tuning lives in this file.
- `modules/team_assigner.py` — bootstraps team K-means from collected torso crops; tracks per-player feature samples and per-team colour families. Re-fit window and assignment margin are configurable.
- `modules/player_ball_assigner.py` — two modes: `distance_gates` (default; foot-point distance + best/second-best margin, with bbox-containment override for chest-high balls) and `foot_overlap` (strict, ball bbox ∩ foot ROI). Stats and overlay use the same assigner so they don't drift.
- `modules/camera_movement_estimator.py` — sparse LK optical flow on left/right border strips (not the centre, where players move). Downscales analysis frames to keep pyramid buffers small on 4K+ inputs.
- `modules/stats/stats_manager.py` — `StatsManager` aggregates `BallVisibilityAnalysis`, `PossessionAnalysis`, `TouchesAnalysis`, `PassEntryAnalysis`. Possession uses a rolling-majority smoothed owner ID (`stats_owner_smoothing_window`); visibility uses raw owner so brief unknown frames don't over-count inferred possession. Output is a JSON payload written under `outputs/stats/`.
- `modules/stats/streamlit_app.py` — the Streamlit UI. Imports `run_analysis` from `main`, so circular-import care is needed when refactoring `main.py`.
- `helpers/foot_roi.py` — foot ROI geometry shared by `player_ball_assigner` and `touches`. Touch detection considers ball ∩ foot ROI plus motion-change heuristics in render config.
- `cache/` holds tracking-pass pickles keyed by params; safe to delete to force re-tracking.

## Performance defaults

When unsure (CPU-first), start with `scale=0.5`, `step=2`, `batch=30` (per the constitution). For GPU runs, `main.py` uses `device="cuda:0"`, `batch=32`, `imgsz=1024`, `use_half=False`, and the Media Foundation H.264 hardware encoder. `render_gpu_pipeline=True` switches to ffmpeg hwaccel decode + torch CUDA overlays + ffmpeg hwaccel encode (see `iter_video_frames_ffmpeg_hwaccel`).
