from functools import lru_cache
from pathlib import Path
import os

from pydantic import BaseModel


def _bool_from_env(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    app_data_dir: Path = Path.home() / "AppData" / "Local" / "Salience"
    database_path: Path | None = None
    demo_mode: bool = False
    demo_data_path: Path = project_root() / "demo-data" / "clips.json"
    static_dir: Path = project_root() / "frontend" / "dist"
    vlm_provider: str = "fireworks"
    student_artifacts_dir: Path = (
        project_root() / ".local-data" / "student" / "artifacts"
    )
    fireworks_api_key: str | None = None
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_model: str = "accounts/fireworks/models/qwen3p7-plus"
    amd_developer_cloud_api_key: str | None = None
    amd_developer_cloud_base_url: str | None = None
    amd_developer_cloud_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    accelerator: str = "auto"
    local_ocr_enabled: bool = True
    label_import_dir: Path | None = None
    windows_videos_prefix: str = "C:\\Users\\YourName\\Videos"
    host_videos_dir: Path | None = None
    trusted_media_dirs: list[Path] = [
        project_root() / "sample-clips",
        project_root() / "demo-video",
    ]

    def resolved_database_path(self) -> Path:
        return self.database_path or self.app_data_dir / "salience.db"


@lru_cache
def get_settings() -> Settings:
    database_path = os.getenv("SALIENCE_DATABASE_PATH")
    return Settings(
        app_data_dir=Path(
            os.getenv("SALIENCE_APP_DATA_DIR", str(Settings().app_data_dir))
        ),
        database_path=Path(database_path) if database_path else None,
        demo_mode=_bool_from_env(os.getenv("SALIENCE_DEMO_MODE")),
        demo_data_path=Path(
            os.getenv("SALIENCE_DEMO_DATA_PATH", str(Settings().demo_data_path))
        ),
        static_dir=Path(os.getenv("SALIENCE_STATIC_DIR", str(Settings().static_dir))),
        vlm_provider=os.getenv("SALIENCE_VLM_PROVIDER", Settings().vlm_provider),
        student_artifacts_dir=Path(
            os.getenv(
                "SALIENCE_STUDENT_ARTIFACTS_DIR",
                str(Settings().student_artifacts_dir),
            )
        ),
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
        fireworks_base_url=os.getenv(
            "FIREWORKS_BASE_URL", Settings().fireworks_base_url
        ),
        fireworks_model=os.getenv("FIREWORKS_MODEL", Settings().fireworks_model),
        amd_developer_cloud_api_key=os.getenv("AMD_DEVELOPER_CLOUD_API_KEY"),
        amd_developer_cloud_base_url=os.getenv("AMD_DEVELOPER_CLOUD_BASE_URL"),
        amd_developer_cloud_model=os.getenv(
            "AMD_DEVELOPER_CLOUD_MODEL", Settings().amd_developer_cloud_model
        ),
        accelerator=os.getenv("SALIENCE_ACCELERATOR", Settings().accelerator),
        local_ocr_enabled=_bool_from_env(
            os.getenv("SALIENCE_LOCAL_OCR_ENABLED", "true")
        ),
        label_import_dir=Path(os.getenv("SALIENCE_LABEL_IMPORT_DIR"))
        if os.getenv("SALIENCE_LABEL_IMPORT_DIR")
        else None,
        windows_videos_prefix=os.getenv(
            "SALIENCE_WINDOWS_VIDEOS_PREFIX", Settings().windows_videos_prefix
        ),
        host_videos_dir=Path(os.getenv("SALIENCE_HOST_VIDEOS_DIR"))
        if os.getenv("SALIENCE_HOST_VIDEOS_DIR")
        else None,
        trusted_media_dirs=[
            Path(value)
            for value in os.getenv(
                "SALIENCE_TRUSTED_MEDIA_DIRS",
                os.pathsep.join(str(path) for path in Settings().trusted_media_dirs),
            ).split(os.pathsep)
            if value
        ],
    )
