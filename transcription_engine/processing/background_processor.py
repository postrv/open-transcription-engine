# File: transcription_engine/processing/background_processor.py
"""Background processing module for handling audio transcription tasks.

Manages a queue of audio processing jobs with status tracking and cleanup.
"""

import asyncio
import logging
import shutil
from collections.abc import AsyncGenerator
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel

from ..audio_input.recorder import AudioLoader
from ..speaker_id.diarizer import DiarizationManager
from ..utils.config import config_manager
from ..whisper_engine.transcriber import WhisperManager

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Status states for processing jobs."""

    QUEUED = "queued"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    IDENTIFYING_SPEAKERS = "identifying_speakers"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANUP = "cleanup"


class ProcessingJob(BaseModel):
    """Represents a single audio processing job."""

    id: UUID
    filename: str
    file_path: Path
    status: ProcessingStatus
    created_at: datetime
    error: str | None = None
    progress: float = 0.0
    output_path: Path | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ProcessingUpdate(BaseModel):
    """Status update for a processing job."""

    job_id: UUID
    status: ProcessingStatus
    progress: float
    error: str | None = None
    output_path: str | None = None


class BackgroundProcessor:
    """Manages background processing of audio files."""

    def __init__(self: "BackgroundProcessor") -> None:
        """Initialize the background processor."""
        self.config = config_manager.load_config()
        self.jobs: dict[UUID, ProcessingJob] = {}
        self.processing_queue: asyncio.Queue[UUID] = asyncio.Queue()
        self.running = False
        self._setup_managers()

    def _setup_managers(self: "BackgroundProcessor") -> None:
        """Initialize transcription and diarization managers."""
        # Initialize WhisperManager
        self.whisper = WhisperManager()
        success = self.whisper.load_model()
        if not success:
            msg = "Failed to load Whisper model"
            raise RuntimeError(msg)

        # Initialize DiarizationManager if enabled
        self.diarizer = (
            DiarizationManager(self.config.diarization)
            if self.config.diarization.enabled
            else None
        )

    async def start(self: "BackgroundProcessor") -> None:
        """Start the background processing loop."""
        if self.running:
            return

        self.running = True
        asyncio.create_task(self._process_queue())
        logger.info("Background processor started")

    async def stop(self: "BackgroundProcessor") -> None:
        """Stop the background processing loop."""
        self.running = False
        # Wait for queue to empty
        if not self.processing_queue.empty():
            await self.processing_queue.join()
        logger.info("Background processor stopped")

    async def submit_job(
        self: "BackgroundProcessor",
        file_path: Path,
    ) -> ProcessingJob:
        """Submit a new audio file for processing.

        Args:
            file_path: Path to the audio file

        Returns:
            ProcessingJob: The created job

        Raises:
            ValueError: If file doesn't exist or invalid format
        """
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise ValueError(msg)

        if file_path.suffix.lower() not in AudioLoader.SUPPORTED_FORMATS:
            msg = f"Unsupported audio format: {file_path.suffix}"
            raise ValueError(msg)

        job_id = uuid4()
        job = ProcessingJob(
            id=job_id,
            filename=file_path.name,
            file_path=file_path,
            status=ProcessingStatus.QUEUED,
            created_at=datetime.now(),
            output_path=self.config.output_dir / f"{job_id}.json",
        )
        self.jobs[job_id] = job
        await self.processing_queue.put(job_id)

        logger.info("Job submitted: %s", job_id)
        return job

    async def get_job_status(
        self: "BackgroundProcessor",
        job_id: UUID,
    ) -> ProcessingJob:
        """Get the current status of a job.

        Args:
            job_id: UUID of the job to check

        Returns:
            ProcessingJob: The job status

        Raises:
            KeyError: If job_id not found
        """
        if job_id not in self.jobs:
            msg = f"Job not found: {job_id}"
            raise KeyError(msg)
        return self.jobs[job_id]

    async def watch_job(
        self: "BackgroundProcessor",
        job_id: UUID,
    ) -> AsyncGenerator[ProcessingUpdate, None]:
        """Watch a job's progress.

        Args:
            job_id: UUID of the job to watch

        Yields:
            ProcessingUpdate: Status updates for the job
        """
        if job_id not in self.jobs:
            msg = f"Job not found: {job_id}"
            raise KeyError(msg)

        job = self.jobs[job_id]
        last_status = job.status
        last_progress = job.progress

        while True:
            # Check if job updated
            if job.status != last_status or abs(job.progress - last_progress) >= 0.01:
                yield ProcessingUpdate(
                    job_id=job_id,
                    status=job.status,
                    progress=job.progress,
                    error=job.error,
                    output_path=str(job.output_path) if job.output_path else None,
                )
                last_status = job.status
                last_progress = job.progress

            # Exit if job completed or failed
            if job.status in {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }:
                break

            await asyncio.sleep(0.1)  # Prevent tight loop

    async def _process_queue(self: "BackgroundProcessor") -> None:
        """Process jobs from the queue."""
        while self.running:
            try:
                job_id = await self.processing_queue.get()
                job = self.jobs[job_id]

                try:
                    # Process the job
                    await self._process_job(job)

                except RuntimeError as e:
                    logger.error("Error processing job %s: %s", job_id, e)

                    job.status = ProcessingStatus.FAILED

                    job.error = str(e)

                finally:
                    self.processing_queue.task_done()
                    # Cleanup if needed
                    if job.status == ProcessingStatus.FAILED:
                        await self._cleanup_job(job)

            except asyncio.CancelledError:
                raise

            except RuntimeError as e:
                logger.error("Error in processing loop: %s", e)

            await asyncio.sleep(1)  # Prevent tight loop on errors

    async def _process_job(
        self: "BackgroundProcessor",
        job: ProcessingJob,
    ) -> None:
        """Process a single job.

        Args:
            job: Job to process
        """
        try:
            # Load audio file
            job.status = ProcessingStatus.LOADING
            job.progress = 0.0
            audio_data, sample_rate = AudioLoader.load_file(job.file_path)

            # Transcribe
            job.status = ProcessingStatus.TRANSCRIBING
            segments = self.whisper.transcribe(
                audio_data,
                sample_rate,
                progress_callback=lambda p: setattr(job, "progress", p * 0.6),
            )

            # Speaker diarization if enabled
            if self.diarizer and self.config.diarization.enabled:
                job.status = ProcessingStatus.IDENTIFYING_SPEAKERS
                job.progress = 60.0

                diarization_segments = self.diarizer.process_singlechannel(
                    audio_data,
                    sample_rate,
                )
                segments = self.diarizer.assign_speaker_ids(
                    segments,
                    diarization_segments,
                )
                job.progress = 90.0

            # Save output
            if job.output_path:
                job.output_path.parent.mkdir(parents=True, exist_ok=True)
                from ..redaction.redactor import TranscriptRedactor

                redactor = TranscriptRedactor()
                # Run auto-redaction
                redacted_segments, _ = redactor.auto_redact(segments)
                redactor.save_redactions(redacted_segments, job.output_path)

            job.status = ProcessingStatus.COMPLETED
            job.progress = 100.0
            logger.info("Job completed successfully: %s", job.id)

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            logger.error("Job failed: %s - %s", job.id, e)
            raise

    async def _cleanup_job(self: "BackgroundProcessor", job: ProcessingJob) -> None:
        """Clean up resources for a job.

        Args:
            job: Job to clean up
        """
        job.status = ProcessingStatus.CLEANUP
        try:
            # Remove temporary files but keep output file
            if self.config.security.audit_logging:
                logger.info(
                    "Preserving files for audit: %s",
                    job.id,
                )
            else:
                temp_dir = self.config.temp_dir / str(job.id)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except (OSError, shutil.Error) as e:
            logger.error("Error cleaning up job %s: %s", job.id, e)

        finally:
            # Keep job record for history
            job.status = (
                ProcessingStatus.COMPLETED
                if job.error is None
                else ProcessingStatus.FAILED
            )
