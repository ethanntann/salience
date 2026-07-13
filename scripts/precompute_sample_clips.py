"""Run the bundled local student over the exact judge sample clips."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi.testclient import TestClient

from salience_api.app import create_app
from salience_api.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()

    clips = args.clips.resolve()
    database = args.database.resolve()
    artifacts = args.artifacts.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        app_data_dir=database.parent,
        database_path=database,
        demo_mode=False,
        vlm_provider="local",
        student_artifacts_dir=artifacts,
        local_ocr_enabled=True,
        trusted_media_dirs=[clips],
        static_dir=Path("__missing_frontend__"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/folders/scan", json={"path": str(clips), "enrich": True}
        )
        response.raise_for_status()
        payload = response.json()
        samples = [
            clip
            for clip in payload["clips"]
            if Path(clip["path"]).parent.resolve() == clips
        ]
        labeled = [clip for clip in samples if clip["teacher_provider"] == "local"]
        described = [clip for clip in labeled if clip["highlight_description"]]
        print(
            f"Indexed {len(samples)}; local labels {len(labeled)}; "
            f"verified descriptions {len(described)}"
        )
        if len(labeled) != len(samples):
            raise RuntimeError("Not every sample clip received a local-student label")


if __name__ == "__main__":
    main()
