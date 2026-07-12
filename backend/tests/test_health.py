from fastapi.testclient import TestClient

from salience_api.app import create_app
from salience_api.config import Settings


def test_health_returns_ok(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path))
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
