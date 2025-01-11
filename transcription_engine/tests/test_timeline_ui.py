# File: transcription_engine/tests/test_timeline_ui.py
"""Tests for the timeline visualization interface."""

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ..timeline_visualization.timeline_ui import RedactionRequest, TimelineUI, app


@pytest.fixture  # type: ignore[misc]
def test_client() -> TestClient:
    """Provide a FastAPI test client."""
    return TestClient(app)


@pytest.fixture  # type: ignore[misc]
def sample_transcript() -> list[dict[str, Any]]:
    """Provide sample transcript data."""
    return [
        {
            "text": "First test segment",
            "start": 0.0,
            "end": 2.0,
            "speaker_id": "SPEAKER_1",
            "confidence": 0.95,
        },
        {
            "text": "Second test segment",
            "start": 2.5,
            "end": 4.5,
            "speaker_id": "SPEAKER_2",
            "confidence": 0.92,
        },
    ]


@pytest_asyncio.fixture  # type: ignore[misc]
async def timeline_ui() -> TimelineUI:
    """Provide TimelineUI instance."""
    return TimelineUI(host="localhost", port=8000)


class TestTimelineUI:
    """Test suite for TimelineUI class."""

    @pytest.mark.asyncio(loop_scope="function")  # type: ignore[misc]
    async def test_load_transcript(
        self: "TestTimelineUI",
        timeline_ui: TimelineUI,
        sample_transcript: list[dict[str, Any]],
    ) -> None:
        """Test loading transcript data."""
        response = await timeline_ui.load_transcript(sample_transcript)
        pytest.assume(response["status"] == "success")
        pytest.assume(response["segments"] == len(sample_transcript))
        pytest.assume(timeline_ui.current_transcript == sample_transcript)

    @pytest.mark.asyncio(loop_scope="function")  # type: ignore[misc]
    async def test_get_transcript(
        self: "TestTimelineUI",
        timeline_ui: TimelineUI,
        sample_transcript: list[dict[str, Any]],
    ) -> None:
        """Test retrieving transcript data."""
        await timeline_ui.load_transcript(sample_transcript)
        transcript = await timeline_ui.get_transcript()
        pytest.assume(len(transcript) == len(sample_transcript))
        pytest.assume(transcript[0]["text"] == sample_transcript[0]["text"])

    @pytest.mark.asyncio(loop_scope="function")  # type: ignore[misc]
    async def test_add_redaction(
        self: "TestTimelineUI",
        timeline_ui: TimelineUI,
        sample_transcript: list[dict[str, Any]],
    ) -> None:
        """Test adding redaction zones."""
        await timeline_ui.load_transcript(sample_transcript)

        redaction = RedactionRequest(
            start_time=0.5,
            end_time=1.5,
            text="test segment",
            reason="Test redaction",
        )

        response = await timeline_ui.add_redaction(redaction)
        pytest.assume(response["status"] == "success")

        transcript = await timeline_ui.get_transcript()
        segment = transcript[0]
        pytest.assume("redaction_zones" in segment)
        pytest.assume(len(segment["redaction_zones"]) == 1)

    def test_save_transcript(
        self: "TestTimelineUI",
        timeline_ui: TimelineUI,
        sample_transcript: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test saving transcript data."""
        # Manual setting of transcript data for sync test
        timeline_ui.current_transcript = sample_transcript
        output_path = tmp_path / "transcript.json"

        timeline_ui.save_transcript(output_path)

        pytest.assume(output_path.exists())
        with open(output_path) as f:
            saved_data = json.load(f)
            pytest.assume("segments" in saved_data)
            pytest.assume("metadata" in saved_data)

    def test_error_handling(self: "TestTimelineUI", timeline_ui: TimelineUI) -> None:
        """Test error handling for invalid operations."""
        with pytest.raises(ValueError, match="No transcript loaded"):
            timeline_ui.save_transcript(Path("test.json"))


def test_api_endpoints(
    test_client: TestClient,
    sample_transcript: list[dict[str, Any]],
) -> None:
    """Test FastAPI endpoints."""
    # Test loading transcript
    response = test_client.post("/api/transcript/load", json=sample_transcript)
    pytest.assume(response.status_code == 200)
    pytest.assume(response.json()["status"] == "success")

    # Test getting transcript
    response = test_client.get("/api/transcript")
    pytest.assume(response.status_code == 200)
    pytest.assume(len(response.json()) == len(sample_transcript))

    # Test adding redaction
    redaction_data = {
        "start_time": 0.5,
        "end_time": 1.5,
        "text": "test segment",
        "reason": "Test redaction",
    }
    response = test_client.post("/api/redaction", json=redaction_data)
    pytest.assume(response.status_code == 200)
    pytest.assume(response.json()["status"] == "success")


def test_serve_html(test_client: TestClient) -> None:
    """Test serving the main HTML template."""
    response = test_client.get("/")
    pytest.assume(response.status_code == 200)
    pytest.assume("text/html" in response.headers["content-type"])
    pytest.assume('<div id="root">' in response.text)
