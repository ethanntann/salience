from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}


def find_clip_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Clip folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Clip path is not a folder: {folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
