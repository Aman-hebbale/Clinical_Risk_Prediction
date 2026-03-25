"""Integration tests for the /health endpoint."""

from fastapi.testclient import TestClient

from src.api.main import app

client: TestClient = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body() -> None:
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
