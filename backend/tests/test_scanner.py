from pathlib import Path

from salience_api.clips.scanner import find_clip_files


def test_find_clip_files_returns_supported_video_files(tmp_path: Path):
    keep = [
        tmp_path / "a.mp4",
        tmp_path / "nested" / "b.mov",
        tmp_path / "nested" / "c.mkv",
    ]
    skip = [
        tmp_path / "notes.txt",
        tmp_path / "image.png",
    ]
    for path in keep + skip:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

    found = find_clip_files(tmp_path)

    assert found == sorted(keep)
