from pathlib import Path

from salience_api.demo_seed import load_demo_clips


def test_load_demo_clips_reads_seed_file(tmp_path: Path):
    seed = tmp_path / "clips.json"
    seed.write_text(
        """
        [
          {
            "filename": "a.mp4",
            "duration_sec": 10,
            "motion_score": 0.5,
            "audio_peak_score": 0.5,
            "silence_ratio": 0.1,
            "tags": ["sniper"],
            "explanation": "demo"
          }
        ]
        """,
        encoding="utf-8",
    )

    clips = load_demo_clips(seed)

    assert clips[0]["filename"] == "a.mp4"
    assert clips[0]["tags"] == ["sniper"]
