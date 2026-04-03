import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from main import run_analysis


class StatsStreamlitApp:
    def __init__(self, stats_root: str = "outputs/stats", inputs_root: str = "inputs"):
        self.stats_root = stats_root
        self.inputs_root = inputs_root

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _stats_files(self):
        pattern = os.path.join(self.stats_root, "*.stats.json")
        return sorted(glob.glob(pattern))

    def _input_video_files(self):
        inputs_dir = Path(self.inputs_root)
        if not inputs_dir.exists():
            return []
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        files = []
        for p in sorted(inputs_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))
        return files

    @staticmethod
    def _sorted_numeric_key_rows(d: dict, key_name: str, value_name: str):
        rows = []
        for k, v in (d or {}).items():
            try:
                sort_k = float(k)
            except Exception:
                sort_k = float("inf")
            rows.append((sort_k, {key_name: str(k), value_name: v}))
        rows.sort(key=lambda x: x[0])
        return [x[1] for x in rows]

    @staticmethod
    def _resolve_path(path_value: str, stats_file_path: str | None = None):
        if not path_value:
            return None

        raw = Path(path_value)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(Path.cwd() / raw)
            # Project root: streamlit_stats.py lives in project root.
            candidates.append(Path(__file__).resolve().parents[2] / raw)
            if stats_file_path:
                candidates.append(Path(stats_file_path).resolve().parent / raw)

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except Exception:
                continue
        return None

    @staticmethod
    def _bgr_to_hex(color_bgr):
        if not color_bgr or len(color_bgr) < 3:
            return "#888888"
        b, g, r = [int(c) for c in color_bgr[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _render_team_header(self, team_key: str, team_colors: dict):
        team_color = team_colors.get(team_key)
        color_hex = self._bgr_to_hex(team_color)
        st.markdown(
            f"<span style='color:{color_hex}; font-weight:700;'>"
            f"{team_key.replace('_', ' ').title()}</span>",
            unsafe_allow_html=True,
        )

    def _delete_analysis_artifacts(self, stats_file_path: str) -> tuple[bool, list[str], list[str]]:
        """
        Delete selected stats file and linked artifacts.
        Returns: (success, deleted_paths, errors)
        """
        deleted = []
        errors = []
        success = True

        try:
            data = self._load_json(stats_file_path)
        except Exception as exc:
            return False, [], [f"Failed to read stats JSON: {exc}"]

        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        output_video_path = meta.get("output_video_path")

        # 1) delete output video (if present)
        video_path = self._resolve_path(output_video_path, stats_file_path=stats_file_path)
        if video_path is not None and video_path.exists():
            try:
                video_path.unlink()
                deleted.append(str(video_path))
            except Exception as exc:
                success = False
                errors.append(f"Failed to delete output video: {exc}")

            # Remove cached web videos created for Streamlit playback.
            cache_dir = video_path.parent / ".streamlit_video_cache"
            if cache_dir.exists():
                try:
                    for p in cache_dir.glob(f"{video_path.stem}_*_web.mp4"):
                        try:
                            p.unlink()
                            deleted.append(str(p))
                        except Exception as exc:
                            success = False
                            errors.append(f"Failed to delete cached preview video '{p}': {exc}")
                except Exception as exc:
                    success = False
                    errors.append(f"Failed to inspect cached preview videos: {exc}")

        # 2) delete stats JSON itself
        stats_path = Path(stats_file_path)
        if stats_path.exists():
            try:
                stats_path.unlink()
                deleted.append(str(stats_path))
            except Exception as exc:
                success = False
                errors.append(f"Failed to delete stats file: {exc}")

        return success, deleted, errors

    @staticmethod
    def _ensure_browser_compatible_video(video_path: Path):
        """
        Create/reuse a browser-friendly MP4 for Streamlit playback.
        Returns a playable path (original or transcoded).
        """
        try:
            if not video_path.exists():
                return None

            # If ffmpeg is missing, we can only try original.
            if shutil.which("ffmpeg") is None:
                return video_path

            cache_dir = video_path.parent / ".streamlit_video_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            src_mtime = int(video_path.stat().st_mtime)
            out_name = f"{video_path.stem}_{src_mtime}_web.mp4"
            out_path = cache_dir / out_name
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return out_path
            return video_path
        except Exception:
            return video_path

    def _render_stats_dashboard(self, data: dict, stats_file_path: str | None = None):
        meta = data.get("meta", {})
        st.caption(f"Video: {meta.get('video_path', '-')}")
        st.caption(f"Output: {meta.get('output_video_path', '-')}")

        output_video_path = meta.get("output_video_path")
        resolved_video_path = self._resolve_path(output_video_path, stats_file_path=stats_file_path)
        if resolved_video_path is not None:
            st.subheader("Output Video")
            try:
                playable_path = self._ensure_browser_compatible_video(resolved_video_path)
                # Using bytes is more reliable across relative path/cwd differences.
                with open(playable_path, "rb") as f:
                    video_bytes = f.read()
                ext = playable_path.suffix.lower()
                video_format = "video/mp4" if ext == ".mp4" else None
                if video_format is not None:
                    st.video(video_bytes, format=video_format)
                else:
                    st.video(video_bytes)
                st.caption(f"Playing: {playable_path}")
            except Exception as exc:
                st.warning(f"Could not load video in browser player: {exc}")
                st.caption(f"Video file path: {resolved_video_path}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Frames", meta.get("total_frames", 0))
        c2.metric("FPS", round(float(meta.get("fps", 0.0)), 2))
        c3.metric(
            "Ball Visibility Ratio",
            round(float(data.get("ball_visibility", {}).get("ball_visibility_ratio", 0.0)), 4),
        )

        team_colors = data.get("team_colors_bgr", {}) or {}
        possession = data.get("possession", {})
        touches = data.get("touches", {})
        pass_entry = data.get("pass_entry", {})

        st.subheader("Core Match Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Visible Ball Frames", possession.get("visible_ball_frames", 0))
        m2.metric("Unknown Possession Frames", possession.get("unknown_owner_frames", 0))

        st.subheader("Team Possession")
        team_pos_rows = self._sorted_numeric_key_rows(
            possession.get("team_possession_seconds", {}),
            key_name="team_id",
            value_name="seconds",
        )
        team_ratio_map = possession.get("team_possession_ratio_on_visible_ball", {}) or {}
        for row in team_pos_rows:
            row["ratio_on_visible_ball"] = round(float(team_ratio_map.get(row["team_id"], 0.0)), 4)
        if team_pos_rows:
            for row in team_pos_rows:
                team_id = str(row.get("team_id"))
                color_hex = self._bgr_to_hex(team_colors.get(team_id))
                st.markdown(
                    (
                        f"<div style='color:{color_hex}; font-weight:700; margin-bottom:0.2rem;'>"
                        f"Team {team_id}: {round(float(row.get('seconds', 0.0)), 2)}s "
                        f"({round(float(row.get('ratio_on_visible_ball', 0.0)) * 100.0, 2)}%)"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            st.dataframe(team_pos_rows, width="stretch")
            chart_data = {r["team_id"]: float(r["seconds"]) for r in team_pos_rows}
            st.bar_chart(chart_data, height=240)
        else:
            st.info("No team possession data available.")

        st.subheader("Touches")
        st.caption(
            "Foot–ball contacts (ball bbox overlaps estimated foot regions), not possession changes."
        )
        touch_rows = self._sorted_numeric_key_rows(
            touches.get("team_touches", {}),
            key_name="team_id",
            value_name="touches",
        )
        if touch_rows:
            for row in touch_rows:
                team_id = str(row.get("team_id"))
                color_hex = self._bgr_to_hex(team_colors.get(team_id))
                st.markdown(
                    (
                        f"<div style='color:{color_hex}; font-weight:700; margin-bottom:0.2rem;'>"
                        f"Team {team_id}: {int(float(row.get('touches', 0)))} touches"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            st.dataframe(touch_rows, width="stretch")
            st.bar_chart({r["team_id"]: float(r["touches"]) for r in touch_rows}, height=220)
        else:
            st.info("No touch data available.")

        st.subheader("Pass Heuristics")
        p1 = st.container()
        attempts = pass_entry.get("team_pass_attempts", {}) or {}
        completed = pass_entry.get("team_pass_completed", {}) or {}
        intercepted = pass_entry.get("team_pass_intercepted", {}) or {}
        keys = sorted(set(list(attempts.keys()) + list(completed.keys()) + list(intercepted.keys())))
        pass_rows = []
        for k in keys:
            a = int(attempts.get(k, 0))
            c = int(completed.get(k, 0))
            i = int(intercepted.get(k, 0))
            rate = (float(c) / float(a)) if a > 0 else 0.0
            pass_rows.append(
                {
                    "team_id": k,
                    "attempts": a,
                    "completed": c,
                    "intercepted": i,
                    "completion_rate": round(rate, 4),
                }
            )
        if pass_rows:
            for row in pass_rows:
                team_id = str(row.get("team_id"))
                color_hex = self._bgr_to_hex(team_colors.get(team_id))
                st.markdown(
                    (
                        f"<div style='color:{color_hex}; font-weight:700; margin-bottom:0.2rem;'>"
                        f"Team {team_id}: {int(float(row.get('completed', 0)))} completed "
                        f"/ {int(float(row.get('attempts', 0)))} attempts "
                        f"({round(float(row.get('completion_rate', 0.0)) * 100.0, 2)}%)"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            p1.dataframe(pass_rows, width="stretch")
            p1.bar_chart({r["team_id"]: float(r["completed"]) for r in pass_rows}, height=220)
        else:
            p1.info("No pass heuristics available.")

    def run(self):
        st.set_page_config(page_title="Football Match Analysis", layout="wide")
        st.title("Football Match Analysis")

        # Sidebar: upload new video and run analysis.
        with st.sidebar:
            st.header("New analysis")
            chosen_input_path = None

            existing_inputs = self._input_video_files()
            if existing_inputs:
                st.subheader("Choose from inputs folder")
                selected_existing = st.selectbox(
                    "Existing input video",
                    options=existing_inputs,
                    index=len(existing_inputs) - 1,
                    key="existing_input_video",
                )
                if selected_existing:
                    chosen_input_path = selected_existing

            st.subheader("Or upload a new video")
            uploaded = st.file_uploader(
                "Upload match video",
                type=["mp4", "avi", "mov", "mkv"],
            )
            run_clicked = False
            saved_input_path = None

            if uploaded is not None:
                Path(self.inputs_root).mkdir(parents=True, exist_ok=True)
                save_path = Path(self.inputs_root) / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.read())
                st.success(f"Saved to {save_path}")
                saved_input_path = str(save_path)
                chosen_input_path = saved_input_path

            if st.button("Run analysis", type="primary"):
                run_clicked = True

            if run_clicked and chosen_input_path:
                with st.spinner("Running analysis... this may take a while."):
                    result = run_analysis(chosen_input_path)
                st.success("Analysis finished.")
                if result.get("input_fps") is not None:
                    st.info(f"Detected input FPS: {float(result['input_fps']):.2f}")
                if result.get("stats_path"):
                    st.info(f"New stats file: {result['stats_path']}")
            elif run_clicked and not chosen_input_path:
                st.warning("Choose an existing input video or upload a new one.")

        files = self._stats_files()
        if not files:
            st.warning(f"No stats files found in: {self.stats_root}")
            return

        # History selection.
        st.subheader("Analysis history")
        selected = st.selectbox("Select analysis", files, index=len(files) - 1)
        col_a, col_b = st.columns([3, 2])
        with col_a:
            confirm_delete = st.checkbox(
                "Confirm delete selected analysis",
                value=False,
                key=f"confirm_delete::{selected}",
            )
        with col_b:
            delete_clicked = st.button(
                "Delete selected analysis",
                type="secondary",
                use_container_width=True,
            )

        if delete_clicked:
            if not confirm_delete:
                st.warning("Tick 'Confirm delete selected analysis' first.")
                st.stop()
            ok, deleted_paths, delete_errors = self._delete_analysis_artifacts(selected)
            if ok:
                st.success("Selected analysis was deleted.")
            else:
                st.warning("Analysis deletion completed with some issues.")
            if deleted_paths:
                with st.expander("Deleted files", expanded=False):
                    for p in deleted_paths:
                        st.write(p)
            if delete_errors:
                for err in delete_errors:
                    st.error(err)
            st.rerun()

        data = self._load_json(selected)

        self._render_stats_dashboard(data, stats_file_path=selected)

