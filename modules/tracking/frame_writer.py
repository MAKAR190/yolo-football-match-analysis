"""
FrameWriter: encapsulates the ffmpeg-pipe + cv2 fallback encoding logic
that both the streaming pipeline and (eventually) the batch render pass use.

Behavior mirrors the inline writer block in render_video_from_tracks:
  - Probe ffmpeg's available encoders, pick the highest-priority one matching
    the configured `hw_encoder` (or "auto" → first available HW encoder).
  - Open an ffmpeg subprocess that reads raw bgr24 frames from stdin and
    encodes to H.264 (with browser-friendly `+faststart` for .mp4).
  - If ffmpeg is unavailable, the chosen encoder fails to start, or the
    pipe breaks mid-run, transparently fall back to cv2.VideoWriter.

Adding new encoders / tweaking quality settings: edit `_encoder_args` below
so both pipelines (streaming + batch) pick up the change in one place.
"""
import os
import re
import subprocess
from typing import Optional

import cv2
import numpy as np

from helpers import find_ffmpeg


# Browser-playable defaults shared with render_video_from_tracks.
_ENCODER_PRIORITY = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libx264"]


def _encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-preset", "medium", "-global_quality", "20", "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        return ["-quality", "balanced", "-rc", "cqp", "-qp_i", "19", "-qp_p", "20", "-pix_fmt", "yuv420p"]
    if encoder == "h264_mf":
        # h264_mf quality 1-100, higher is better. ~85 is high-quality on motion content.
        return ["-rate_control", "quality", "-quality", "85", "-pix_fmt", "yuv420p"]
    if encoder == "libx264":
        return ["-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"]
    # Unknown encoder: assume libx264-style flags.
    return ["-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"]


def _fallback_fourcc_for_ext(ext: str, preferred_codec: str) -> str:
    # OpenCV+MP4 only supports a subset of fourccs; avoid MJPG/XVID in .mp4.
    if ext == ".mp4":
        if preferred_codec.upper() in ("XVID", "MJPG", "MJPEG"):
            return "mp4v"
    return preferred_codec


class FrameWriter:
    """
    Writes BGR uint8 frames to disk via ffmpeg HW encode (preferred) or
    cv2.VideoWriter (fallback). Same shape (H, W, 3) for every write.

    Usage:
        w = FrameWriter(path, width, height, fps, render_cfg)
        for frame in frames:
            w.write(frame)
        w.close()
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        render_cfg,  # RenderOwnershipStatsConfig (avoid circular import)
    ):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.cfg = render_cfg
        self._output_ext = os.path.splitext(output_path)[1].lower()

        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_stdin = None
        self._cv2_writer: Optional[cv2.VideoWriter] = None
        self._selected_encoder: Optional[str] = None

        self._open()

    # ---- setup ----

    def _open(self):
        if self.cfg.use_hw_encode:
            self._try_open_ffmpeg()
        if self._ffmpeg_stdin is None:
            self._open_cv2_fallback()

    def _try_open_ffmpeg(self):
        ffmpeg_exe = find_ffmpeg()
        if ffmpeg_exe is None:
            return

        encoders_text = ""
        try:
            probe = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            encoders_text = (probe.stdout or "") + (probe.stderr or "")
        except Exception:
            return

        def has_encoder(name: str) -> bool:
            return re.search(rf"\b{re.escape(name)}\b", encoders_text) is not None

        if self.cfg.hw_encoder != "auto":
            if has_encoder(self.cfg.hw_encoder):
                self._selected_encoder = self.cfg.hw_encoder
        else:
            for candidate in _ENCODER_PRIORITY:
                if has_encoder(candidate):
                    self._selected_encoder = candidate
                    break
        if self._selected_encoder is None:
            return

        cmd = [
            ffmpeg_exe, "-nostdin", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps}",
            "-i", "-",
            "-an",
            "-vcodec", self._selected_encoder,
        ]
        cmd += _encoder_args(self._selected_encoder)
        if self._output_ext == ".mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += [self.output_path]

        try:
            # stderr must NOT be PIPE without a reader: ffmpeg fills the buffer
            # and blocks, then stdin breaks with BrokenPipeError.
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._ffmpeg_stdin = self._ffmpeg_proc.stdin
            print(f"[FrameWriter] Encoding via ffmpeg -vcodec {self._selected_encoder}")
        except Exception:
            self._ffmpeg_proc = None
            self._ffmpeg_stdin = None

    def _open_cv2_fallback(self):
        codec_name = _fallback_fourcc_for_ext(self._output_ext, self.cfg.codec)
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        self._cv2_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if self._selected_encoder is None:
            print(f"[FrameWriter] Encoding via cv2.VideoWriter fourcc={codec_name}")

    # ---- write ----

    def write(self, frame: np.ndarray):
        if self._ffmpeg_stdin is not None:
            try:
                self._ffmpeg_stdin.write(frame.tobytes())
                return
            except (BrokenPipeError, OSError):
                # HW encoder failed mid-run: shut down ffmpeg cleanly and
                # transparently switch to cv2.VideoWriter.
                self._teardown_ffmpeg()
                if self._cv2_writer is None:
                    self._open_cv2_fallback()
                print("[FrameWriter] Warning: ffmpeg pipe broke; falling back to cv2.VideoWriter")
        if self._cv2_writer is not None:
            self._cv2_writer.write(frame)

    def close(self):
        self._teardown_ffmpeg()
        if self._cv2_writer is not None:
            try:
                self._cv2_writer.release()
            except Exception:
                pass
            self._cv2_writer = None

    def _teardown_ffmpeg(self):
        try:
            if self._ffmpeg_stdin is not None:
                self._ffmpeg_stdin.close()
        except Exception:
            pass
        self._ffmpeg_stdin = None
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.wait(timeout=10)
            except Exception:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass
        self._ffmpeg_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
