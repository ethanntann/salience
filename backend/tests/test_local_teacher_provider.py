from pathlib import Path
import json

from salience_api.app import build_teacher_client, teacher_provider_configured
from salience_api.config import Settings
from salience_api.student.local_teacher import LocalTeacherClient
from salience_api.student.backbone import ARTIFACT_VERSION


def _write_artifact_files(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "locator.onnx").write_bytes(b"fake-locator")
    (artifacts_dir / "event_heads.onnx").write_bytes(b"fake-event-heads")
    (artifacts_dir / "artifact_meta.json").write_text(
        json.dumps({"version": ARTIFACT_VERSION}), encoding="utf-8"
    )
    (artifacts_dir / "thresholds.json").write_text(
        json.dumps({"weapon": 0.6}), encoding="utf-8"
    )


def test_teacher_provider_configured_requires_local_artifact_files(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    settings = Settings(
        app_data_dir=tmp_path,
        vlm_provider="local",
        student_artifacts_dir=artifacts_dir,
    )

    assert teacher_provider_configured(settings) is False

    _write_artifact_files(artifacts_dir)

    assert teacher_provider_configured(settings) is True


def test_teacher_client_returns_local_teacher_client(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifact_files(artifacts_dir)
    settings = Settings(
        app_data_dir=tmp_path,
        vlm_provider="local",
        student_artifacts_dir=artifacts_dir,
    )
    captured: dict[str, Path] = {}

    def fake_from_artifacts(
        path: Path, *, accelerator: str = "auto"
    ) -> LocalTeacherClient:
        captured["artifacts_dir"] = path
        return LocalTeacherClient.from_sessions(
            models=object(),  # type: ignore[arg-type]
            model="test-local-student",
        )

    monkeypatch.setattr(
        "salience_api.app.LocalTeacherClient.from_artifacts",
        fake_from_artifacts,
    )

    client = build_teacher_client(settings)

    assert isinstance(client, LocalTeacherClient)
    assert client.model == "test-local-student"
    assert captured["artifacts_dir"] == artifacts_dir
