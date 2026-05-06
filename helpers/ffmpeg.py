import shutil


def find_ffmpeg() -> str | None:
    """Return a path to an ffmpeg executable, or None if unavailable.

    Prefers a system ffmpeg on PATH; falls back to the binary bundled with
    `imageio-ffmpeg` so the H.264 encode/transcode paths work on machines
    where ffmpeg isn't installed system-wide.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
