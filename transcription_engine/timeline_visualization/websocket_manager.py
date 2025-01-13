# File: transcription_engine/timeline_visualization/websocket_manager.py
"""WebSocket manager for real-time processing updates.

Handles WebSocket connections and broadcasting of processing status updates.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..processing.background_processor import ProcessingStatus, ProcessingUpdate

# Configure logging
logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""

    type: str
    data: dict[str, Any]


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self: "ConnectionManager") -> None:
        """Initialize the connection manager."""
        # Store active connections for each job
        self.active_connections: dict[UUID, list[WebSocket]] = {}
        # Lock for thread-safe connection management
        self._lock = asyncio.Lock()

    async def connect(
        self: "ConnectionManager", websocket: WebSocket, job_id: UUID
    ) -> None:
        """Accept and store a new WebSocket connection.

        Args:
            websocket: WebSocket connection to manage
            job_id: UUID of the job to watch

        Raises:
            RuntimeError: If accepting the connection fails
        """
        try:
            await websocket.accept()
            async with self._lock:
                if job_id not in self.active_connections:
                    self.active_connections[job_id] = []
                self.active_connections[job_id].append(websocket)
                logger.info(
                    "New WebSocket connection for job %s (total: %d)",
                    job_id,
                    len(self.active_connections[job_id]),
                )
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.error("Error accepting WebSocket connection: %s", e)
            error_message = f"Failed to establish WebSocket connection: {e}"
            raise RuntimeError(error_message) from e

    async def disconnect(
        self: "ConnectionManager", websocket: WebSocket, job_id: UUID
    ) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
            job_id: UUID of the job
        """
        async with self._lock:
            if job_id in self.active_connections:
                try:
                    self.active_connections[job_id].remove(websocket)
                    if not self.active_connections[job_id]:
                        del self.active_connections[job_id]
                    logger.info(
                        "WebSocket disconnected for job %s (remaining: %d)",
                        job_id,
                        len(self.active_connections.get(job_id, [])),
                    )
                except ValueError:
                    logger.debug("Connection already removed for job %s", job_id)
                finally:
                    try:
                        await websocket.close()
                    except OSError as e:
                        logger.debug("Error closing websocket: %s", e)

    async def broadcast_to_job(
        self: "ConnectionManager",
        job_id: UUID,
        message: WebSocketMessage,
    ) -> None:
        """Broadcast a message to all connections for a job.

        Args:
            job_id: UUID of the job
            message: Message to broadcast
        """
        if job_id not in self.active_connections:
            return

        # Convert message to JSON once for all connections
        try:
            message_json = message.model_dump_json()
        except ValueError as e:
            logger.error("Error serializing message: %s", e)
            return

        disconnect_tasks = []
        async with self._lock:
            connections = self.active_connections[job_id].copy()

        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except (WebSocketDisconnect, RuntimeError, ValueError) as e:
                logger.warning(
                    "Error sending message to job %s: %s",
                    job_id,
                    e,
                )
                disconnect_tasks.append(self.disconnect(websocket, job_id))

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    def get_connection_count(self: "ConnectionManager", job_id: UUID) -> int:
        """Get the number of active connections for a job.

        Args:
            job_id: UUID of the job

        Returns:
            int: Number of active connections
        """
        return len(self.active_connections.get(job_id, []))


class WebSocketManager:
    """Manages WebSocket endpoints and processing updates."""

    def __init__(self: "WebSocketManager") -> None:
        """Initialize the WebSocket manager."""
        self.connection_manager = ConnectionManager()

    async def handle_job_updates(
        self: "WebSocketManager",
        websocket: WebSocket,
        job_id: UUID,
        update_generator: AsyncGenerator[ProcessingUpdate, None],
    ) -> None:
        """Handle updates for a specific job.

        Args:
            websocket: WebSocket connection
            job_id: UUID of the job to watch
            update_generator: Generator of job updates
        """
        from ..timeline_visualization.timeline_ui import background_processor

        current_job = None
        try:
            # Get current job status before accepting connection
            current_job = await background_processor.get_job_status(job_id)

            # For completed/failed jobs, send final status and close cleanly
            if current_job.status in {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }:
                await websocket.accept()
                message = WebSocketMessage(
                    type="processing_update",
                    data={
                        "job_id": str(job_id),
                        "status": current_job.status,
                        "progress": 100,
                        "error": current_job.error,
                        "output_path": str(current_job.output_path)
                        if current_job.output_path
                        else None,
                    },
                )
                await websocket.send_text(message.model_dump_json())
                await websocket.close(1000, "Job already completed")
                return

            # For active jobs, proceed with normal connection handling
            await self.connection_manager.connect(websocket, job_id)
            logger.info("Started handling updates for job %s", job_id)

            async for update in update_generator:
                try:
                    # Convert update to WebSocket message
                    message = WebSocketMessage(
                        type="processing_update",
                        data={
                            "job_id": str(update.job_id),
                            "status": update.status,
                            "progress": update.progress,
                            "error": update.error,
                            "output_path": update.output_path,
                        },
                    )

                    # Broadcast update to all connections for this job
                    await self.connection_manager.broadcast_to_job(job_id, message)
                    logger.debug(
                        "Broadcast update for job %s: %s%%",
                        job_id,
                        update.progress,
                    )

                    # Break if job completed or failed
                    if update.status in {
                        ProcessingStatus.COMPLETED,
                        ProcessingStatus.FAILED,
                    }:
                        logger.info(
                            "Job %s finished with status: %s",
                            job_id,
                            update.status,
                        )
                        break

                except (ValueError, RuntimeError) as e:
                    logger.error("Error processing update for job %s: %s", job_id, e)
                    error_message = WebSocketMessage(
                        type="error",
                        data={"message": f"Update processing error: {str(e)}"},
                    )
                    try:
                        await websocket.send_text(error_message.model_dump_json())
                    except (ValueError, RuntimeError) as e:
                        logger.error("Error sending error message to client: %s", e)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected from job %s", job_id)
        except ValueError as e:
            logger.error("Error handling job %s updates: %s", job_id, e)
            try:
                error_message = WebSocketMessage(
                    type="error",
                    data={"message": str(e)},
                )
                await websocket.send_text(error_message.model_dump_json())
            except (ValueError, RuntimeError):
                logger.error("Failed to send error message to client")
        finally:
            # Only call disconnect if we connected (i.e., job wasn't already complete)
            if current_job is None or current_job.status not in {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }:
                logger.info("Cleaning up connection for job %s", job_id)
                await self.connection_manager.disconnect(websocket, job_id)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
