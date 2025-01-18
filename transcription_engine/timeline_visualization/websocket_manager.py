# File: transcription_engine/timeline_visualization/websocket_manager.py
"""WebSocket manager for real-time processing updates.

Handles WebSocket connections and broadcasting of processing status updates.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..processing.background_processor import ProcessingStatus, ProcessingUpdate

# Configure logging
logger = logging.getLogger(__name__)


# Define WebSocket states since FastAPI doesn't expose them directly
class WebSocketState(str, Enum):
    """WebSocket connection states."""

    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""

    type: str
    data: dict[str, Any]


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self: "ConnectionManager") -> None:
        """Initialize the connection manager."""
        self.active_connections: dict[UUID, list[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # Track connection health
        self.connection_statuses: dict[UUID, dict[WebSocket, bool]] = {}

    async def connect(
        self: "ConnectionManager", websocket: WebSocket, job_id: UUID
    ) -> None:
        """Accept and store a new WebSocket connection."""
        try:
            await websocket.accept()
            async with self._lock:
                if job_id not in self.active_connections:
                    self.active_connections[job_id] = []
                    self.connection_statuses[job_id] = {}

                self.active_connections[job_id].append(websocket)
                self.connection_statuses[job_id][websocket] = True

                logger.info(
                    "New WebSocket connection for job %s (total: %d)",
                    job_id,
                    len(self.active_connections[job_id]),
                )

        except (WebSocketDisconnect, RuntimeError) as e:
            logger.error("Error accepting WebSocket connection: %s", e)

            # Ensure connection is closed on error

            try:
                await websocket.close(code=1011, reason=str(e))

            except (WebSocketDisconnect, RuntimeError) as e:
                logger.error("Error closing websocket: %s", e)

                error_message = f"Failed to establish WebSocket connection: {e}"

                logger.error(error_message)

                raise RuntimeError(error_message) from e

    async def disconnect(
        self: "ConnectionManager",
        websocket: WebSocket,
        job_id: UUID,
        code: int = 1000,
        reason: str = "Normal closure",
    ) -> None:
        """Remove a WebSocket connection with proper cleanup."""
        async with self._lock:
            if job_id in self.active_connections:
                try:
                    # Mark connection as inactive first
                    if job_id in self.connection_statuses:
                        self.connection_statuses[job_id].pop(websocket, None)

                    self.active_connections[job_id].remove(websocket)

                    # Clean up empty job entries
                    if not self.active_connections[job_id]:
                        del self.active_connections[job_id]
                        self.connection_statuses.pop(job_id, None)

                    logger.info(
                        "WebSocket disconnected for job %s (remaining: %d)",
                        job_id,
                        len(self.active_connections.get(job_id, [])),
                    )
                except ValueError:
                    logger.debug("Connection already removed for job %s", job_id)
                finally:
                    try:
                        await websocket.close(code=code, reason=reason)
                    except WebSocketDisconnect as e:
                        logger.debug("WebSocket disconnected: %s", e)
                    except RuntimeError as e:
                        logger.debug("Runtime error closing websocket: %s", e)

    async def broadcast_to_job(
        self: "ConnectionManager",
        job_id: UUID,
        message: WebSocketMessage,
    ) -> None:
        """Broadcast message to all connections for a job w improved error handling."""
        if job_id not in self.active_connections:
            return

        try:
            message_json = message.model_dump_json()
        except ValueError as e:
            logger.error("Error serializing message: %s", e)
            return

        disconnect_tasks = []
        async with self._lock:
            # Only broadcast to connections marked as active
            connections = [
                ws
                for ws in self.active_connections[job_id]
                if self.connection_statuses.get(job_id, {}).get(ws, False)
            ]

        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.warning(
                    "Error broadcasting to job %s: %s",
                    job_id,
                    e,
                )
                disconnect_tasks.append(
                    self.disconnect(websocket, job_id, code=1011, reason=str(e))
                )

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)


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
        """Handle updates for a specific job with improved error handling."""
        from ..timeline_visualization.timeline_ui import background_processor

        current_job = None
        try:
            current_job = await background_processor.get_job_status(job_id)
            logger.info(f"Got job status for {job_id}: {current_job.status}")

            # For completed/failed jobs, send completion status right away
            if current_job.status in {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }:
                await websocket.accept()
                logger.info(f"Sending immediate completion for job {job_id}")
                message = WebSocketMessage(
                    type="processing_update",
                    data={
                        "job_id": str(job_id),
                        "status": current_job.status,
                        "progress": 100.0,
                        "error": current_job.error,
                        "output_path": str(current_job.output_path)
                        if current_job.output_path
                        else None,
                    },
                )
                # Log what we're sending
                logger.info(f"Sending completion message: {message.model_dump_json()}")
                await websocket.send_text(message.model_dump_json())
                await websocket.close(1000, "Job already completed")
                return

            # For active jobs
            await self.connection_manager.connect(websocket, job_id)
            logger.info(f"Started handling updates for job {job_id}")

            async for update in update_generator:
                try:
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

                    # Log each update
                    logger.info(
                        f"Broadcasting update for {job_id}: {message.model_dump_json()}"
                    )
                    await self.connection_manager.broadcast_to_job(job_id, message)

                    if update.status in {
                        ProcessingStatus.COMPLETED,
                        ProcessingStatus.FAILED,
                    }:
                        logger.info(
                            f"Job {job_id} finished with status: {update.status}"
                        )
                        # Ensure final message is sent before disconnecting
                        final_message = WebSocketMessage(
                            type="processing_update",
                            data={
                                "job_id": str(update.job_id),
                                "status": update.status,
                                "progress": 100.0,
                                "error": update.error,
                                "output_path": update.output_path,
                            },
                        )
                        await websocket.send_text(final_message.model_dump_json())
                        await self.connection_manager.disconnect(
                            websocket,
                            job_id,
                            code=1000,
                            reason=f"Job {update.status.lower()}",
                        )
                        break

                except (WebSocketDisconnect, RuntimeError, ValueError) as e:
                    logger.error(f"Error processing update for {job_id}: {e}")
                    try:
                        error_message = WebSocketMessage(
                            type="error",
                            data={"message": f"Update error: {str(e)}"},
                        )
                        await websocket.send_text(error_message.model_dump_json())
                    except (WebSocketDisconnect, RuntimeError, ValueError) as e:
                        logger.error(f"Failed to send error message: {e}")
                        break

        except WebSocketDisconnect:
            logger.info(f"Client disconnected from job {job_id}")
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error handling job {job_id}: {e}")
            if websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    error_message = WebSocketMessage(
                        type="error",
                        data={"message": str(e)},
                    )
                    await websocket.send_text(error_message.model_dump_json())
                except (WebSocketDisconnect, RuntimeError, ValueError) as e:
                    logger.error(f"Failed to send error message: {e}")
        finally:
            if current_job is None or current_job.status not in {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }:
                logger.info(f"Cleaning up connection for job {job_id}")
                await self.connection_manager.disconnect(websocket, job_id)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
