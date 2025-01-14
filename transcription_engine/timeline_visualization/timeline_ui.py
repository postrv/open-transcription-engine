# File: transcription_engine/timeline_visualization/timeline_ui.py
"""Timeline Visualization UI module for reviewing and editing transcripts.

Provides a web interface using FastAPI and React for transcript visualization.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..processing.background_processor import BackgroundProcessor
from ..utils.config import config_manager
from .websocket_manager import websocket_manager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app and processing components
app = FastAPI(title="Transcript Timeline Viewer")
background_processor = BackgroundProcessor()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Type alias for transcript data
TranscriptData = list[dict[str, Any]]

# Global state for transcript data
# Note: In production, use proper state management
_current_transcript: TranscriptData | None = None


class RedactionRequest(BaseModel):
    """Model for redaction requests from the UI."""

    start_time: float
    end_time: float
    text: str
    reason: str


@app.get("/", response_class=HTMLResponse)  # type: ignore
async def serve_html() -> HTMLResponse:
    """Serve the main HTML template.

    Returns:
        HTML template for the timeline viewer
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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


@app.post("/api/transcript/load", response_model=dict[str, Any])  # type: ignore
async def load_transcript_endpoint(transcript_data: TranscriptData) -> dict[str, Any]:
    """Load transcript data into the UI.

    Args:
        transcript_data: List of transcript segments to load

    Returns:
        Dictionary containing operation status and segment count

    Raises:
        HTTPException: If transcript data is invalid
    """
    error_msg = "Empty transcript data"
    if not transcript_data:
        raise HTTPException(status_code=400, detail=error_msg)

    global _current_transcript
    _current_transcript = transcript_data
    return {"status": "success", "segments": len(transcript_data)}


@app.get("/api/transcript/{job_id}", response_model=list[dict[str, Any]])  # type: ignore
async def get_job_transcript_endpoint(job_id: str) -> list[dict[str, Any]]:
    """Get transcript data for a specific job.

    Args:
        job_id: ID of the job to fetch transcript for

    Returns:
        List of transcript segments

    Raises:
        HTTPException: If transcript file cannot be read or parsed
    """
    try:
        # Clean the job_id to remove any .json extension
        job_id = job_id.replace(".json", "")

        # Look for specific job's transcript
        config = config_manager.load_config()
        transcript_file = config.output_dir / f"{job_id}.json"

        if not transcript_file.exists():
            logger.error("Transcript file not found: %s", transcript_file)
            raise HTTPException(
                status_code=404, detail=f"Transcript not found for job {job_id}"
            )

        try:
            with open(transcript_file) as f:
                data: dict[str, Any] = json.load(f)
                segments: list[dict[str, Any]] = data.get("segments", [])
                logger.info("Successfully loaded transcript for job %s", job_id)
                return segments
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse transcript file: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to parse transcript file: {str(e)}"
            ) from e

    except OSError as e:
        logger.error("Error accessing transcript file: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error accessing transcript file: {str(e)}"
        ) from e
    except Exception as general_exception:  # noqa: BLE001
        logger.error("Unexpected error loading transcript: %s", general_exception)
        raise HTTPException(
            status_code=500, detail="Internal server error while loading transcript"
        ) from general_exception


@app.get("/api/transcript", response_model=list[dict[str, Any]])  # type: ignore[misc]
async def get_transcript_endpoint() -> list[dict[str, Any]]:
    """Get the current transcript data.

    Returns:
        List of transcript segments

    Raises:
        HTTPException: If transcript file cannot be read or parsed
    """
    if _current_transcript is None:
        return []
    return _current_transcript


@app.post("/api/redaction", response_model=dict[str, str])  # type: ignore
async def add_redaction_endpoint(redaction: RedactionRequest) -> dict[str, str]:
    """Add a new redaction zone to the transcript.

    Args:
        redaction: RedactionRequest object containing redaction details

    Returns:
        Dictionary containing operation status

    Raises:
        HTTPException: If no transcript is loaded or no matching segment found
    """
    transcript_error = "No transcript loaded"
    if _current_transcript is None:
        raise HTTPException(status_code=404, detail=transcript_error)

    for segment in _current_transcript:
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

    segment_error = "No matching segment found"
    raise HTTPException(status_code=400, detail=segment_error)


# First, mount the static files properly with correct paths
try:
    # Mount the uploads directory first (more specific path)
    app.mount(
        "/uploads",
        StaticFiles(directory="transcription_engine/static/uploads"),
        name="uploads",
    )

    # Mount the main static files (built React app) - this should serve from dist
    app.mount(
        "/static",
        StaticFiles(directory="transcription_engine/static/dist"),
        name="static",
    )
except RuntimeError as e:
    logger.warning("Static files directory not found: %s", e)


@app.websocket("/ws/jobs/{job_id}")  # type: ignore
async def websocket_job_endpoint(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for job status updates.

    Args:
        websocket: WebSocket connection
        job_id: UUID of the job to watch
    """
    try:
        job_uuid = UUID(job_id)
        # Verify job exists (will raise KeyError if not found)
        await background_processor.get_job_status(job_uuid)
        update_generator = background_processor.watch_job(job_uuid)
        await websocket_manager.handle_job_updates(
            websocket,
            job_uuid,
            update_generator,
        )
    except ValueError:
        await websocket.close(code=4000, reason="Invalid job ID")
    except KeyError:
        await websocket.close(code=4004, reason="Job not found")


# Add startup and shutdown events for background processor
@app.on_event("startup")  # type: ignore
async def startup_event() -> None:
    """Start background processor on server startup."""
    await background_processor.start()


@app.on_event("shutdown")  # type: ignore
async def shutdown_event() -> None:
    """Stop background processor on server shutdown."""
    await background_processor.stop()


@app.post("/api/upload-audio")  # type: ignore
async def upload_audio(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Handle audio file upload and start processing.

    Args:
        file: Uploaded audio file

    Returns:
        Dictionary containing job ID and file URL

    Raises:
        HTTPException: If there's an error during upload or processing
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("transcription_engine/static/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        file_extension = Path(file.filename).suffix if file.filename else ".mp3"
        safe_filename = f"audio_{int(time.time())}{file_extension}"
        file_path = uploads_dir / safe_filename

        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Submit processing job
        job = await background_processor.submit_job(file_path)

        # Return both the job ID and file URL
        return {
            "job_id": str(job.id),
            "url": f"/uploads/{safe_filename}",
            "status": job.status,
        }

    except (OSError, ValueError) as e:
        logger.error("Error uploading audio: %s", e)
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


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

        Raises:
            ValueError: If transcript data is invalid
        """
        error_msg = "Empty transcript data"
        if not transcript_data:
            raise ValueError(error_msg)

        # Update both instance and module state
        self.current_transcript = transcript_data
        global _current_transcript
        _current_transcript = transcript_data

        return {"status": "success", "segments": len(transcript_data)}

    async def get_transcript(self: "TimelineUI") -> TranscriptData:
        """Get the current transcript data.

        Returns:
            List of transcript segments

        Raises:
            ValueError: If no transcript is loaded
        """
        error_msg = "No transcript loaded"
        if self.current_transcript is None:
            raise ValueError(error_msg)
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
            ValueError: If no transcript is loaded or no matching segment found
        """
        transcript_error = "No transcript loaded"
        if self.current_transcript is None:
            raise ValueError(transcript_error)

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

        segment_error = "No matching segment found"
        raise ValueError(segment_error)

    def save_transcript(self: "TimelineUI", output_path: Path) -> None:
        """Save the current transcript state to file.

        Args:
            output_path: Path where to save the JSON file

        Raises:
            ValueError: If no transcript is loaded
            OSError: If there are filesystem-related errors
            json.JSONDecodeError: If there are JSON serialization errors
        """
        error_msg = "No transcript loaded"
        if self.current_transcript is None:
            raise ValueError(error_msg)

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
            logger.info(
                "Starting Timeline UI server at http://%s:%s",
                self.host,
                self.port,
            )
            import uvicorn

            uvicorn.run(app, host=self.host, port=self.port)
        except Exception as e:
            logger.error("Error starting Timeline UI server: %s", e)
            raise

    def __del__(self: "TimelineUI") -> None:
        """Cleanup when the object is deleted."""
        logger.info("Shutting down Timeline UI server")
