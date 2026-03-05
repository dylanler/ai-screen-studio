import asyncio

from fastapi.testclient import TestClient

from instruction_video_generator import webapp


def setup_function() -> None:
    webapp.JOBS.clear()


def test_create_job_returns_queue(monkeypatch):
    async def fake_run_job(job_id: str, payload):
        await asyncio.sleep(0)

    monkeypatch.setattr(webapp, "_run_job", fake_run_job)

    with TestClient(webapp.app) as client:
        response = client.post("/api/jobs", json={"prompt": "Show me how to create a table in Google Docs"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["prompt"] == "Show me how to create a table in Google Docs"
    assert len(payload["queue"]) == 5
    assert payload["queue"][0]["step_id"] == "queued"


def test_get_missing_job_returns_404():
    with TestClient(webapp.app) as client:
        response = client.get("/api/jobs/missing-id")

    assert response.status_code == 404
