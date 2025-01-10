# File: transcription_engine/timeline_visualization/timeline_ui.py
"""Timeline Visualization UI module for reviewing and editing transcripts.

Provides a web interface using FastAPI and React for transcript visualization.
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Transcript Timeline Viewer")
router = APIRouter()


# Type alias for transcript data
TranscriptData = list[dict[str, Any]]


class RedactionRequest(BaseModel):
    """Model for redaction requests from the UI."""

    start_time: float
    end_time: float
    text: str
    reason: str


class TimelineUI:
    """Manages the timeline visualization interface."""

    def __init__(self: "TimelineUI", host: str = "localhost", port: int = 8000) -> None:
        """Initialize the timeline UI server.

        Args:
            host: Hostname to bind the server to
            port: Port number to listen on
        """
        self.host = host
        self.port = port
        self.current_transcript: TranscriptData | None = None

    async def load_transcript(
        self: "TimelineUI",
        transcript_data: TranscriptData,
    ) -> dict[str, Any]:
        """Load transcript data into the UI.

        Args:
            transcript_data: List of transcript segments to load

        Returns:
            Dictionary containing status and segment count
        """
        self.current_transcript = transcript_data
        return {"status": "success", "segments": len(transcript_data)}

    async def get_transcript(self: "TimelineUI") -> TranscriptData:
        """Get the current transcript data.

        Returns:
            List of transcript segments

        Raises:
            HTTPException: If no transcript is loaded
        """
        if not self.current_transcript:
            msg = "No transcript loaded"
            raise HTTPException(status_code=404, detail=msg)
        return self.current_transcript

    async def add_redaction(
        self: "TimelineUI",
        redaction: RedactionRequest,
    ) -> dict[str, str]:
        """Add a new redaction zone to the transcript.

        Args:
            redaction: RedactionRequest object containing redaction details

        Returns:
            Dictionary containing operation status

        Raises:
            HTTPException: If no transcript is loaded or no matching segment found
        """
        if not self.current_transcript:
            msg = "No transcript loaded"
            raise HTTPException(status_code=404, detail=msg)

        # Find the relevant segment
        for segment in self.current_transcript:
            if (
                segment["start"] <= redaction.start_time
                and segment["end"] >= redaction.end_time
            ):
                if "redaction_zones" not in segment:
                    segment["redaction_zones"] = []
                segment["redaction_zones"].append(
                    {
                        "start_time": redaction.start_time,
                        "end_time": redaction.end_time,
                        "reason": redaction.reason,
                        "redaction_type": "manual",
                        "confidence": 1.0,
                    },
                )
                return {"status": "success"}

        msg = "No matching segment found"
        raise HTTPException(status_code=400, detail=msg)

    def save_transcript(self: "TimelineUI", output_path: Path) -> None:
        """Save the current transcript state to file.

        Args:
            output_path: Path where to save the JSON file

        Raises:
            ValueError: If no transcript is loaded
            OSError: If there are filesystem-related errors
            json.JSONDecodeError: If there are JSON serialization errors
        """
        if not self.current_transcript:
            msg = "No transcript loaded"
            raise ValueError(msg)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "total_segments": len(self.current_transcript),
                "redacted_segments": len(
                    [s for s in self.current_transcript if s.get("redaction_zones")],
                ),
            }
            output_data = {"segments": self.current_transcript, "metadata": metadata}

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info("Saved transcript to %s", output_path)
        except Exception as e:
            logger.error("Error saving transcript: %s", e)
            raise

    def run(self: "TimelineUI") -> None:
        """Start the timeline UI server."""
        try:
            app.post("/api/transcript/load")(self.load_transcript)
            app.get("/api/transcript")(self.get_transcript)
            app.post("/api/redaction")(self.add_redaction)

            app.mount("/static", StaticFiles(directory="static"), name="static")

            # Add type annotation for the decorator
            @router.get("/{full_path:path}", response_class=HTMLResponse)  # type: ignore
            async def serve_html(full_path: str) -> HTMLResponse:
                """Serve the main HTML template."""
                html_content = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width,
                     initial-scale=1.0">
                    <title>Transcript Timeline</title>
                    <link href="/static/styles.css" rel="stylesheet">
                </head>
                <body>
                    <div id="root"></div>
                    <script src="/static/bundle.js"></script>
                </body>
                </html>
                """
                return HTMLResponse(content=html_content)

            app.include_router(router)

            # Start the server
            logger.info(
                "Starting Timeline UI server at http://%s:%s",
                self.host,
                self.port,
            )
            import uvicorn

            uvicorn.run(app, host=self.host, port=self.port)

        except Exception as e:  # Possibly narrower or # noqa: BLE001
            logger.error("Error starting Timeline UI server: %s", e)

            raise

    def __del__(self: "TimelineUI") -> None:
        """Cleanup when the object is deleted."""
        logger.info("Shutting down Timeline UI server")
