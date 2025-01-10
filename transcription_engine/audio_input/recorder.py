# File: transcription_engine/audio_input/recorder.py
"""Audio Recorder module for the Open Transcription Engine.

Handles both real-time audio capture and file loading with support for
multi-channel audio commonly found in courtroom settings.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import numpy as np
import pyaudio
import soundfile as sf
from soundfile import SoundFileError

from ..utils.config import AudioConfig, config_manager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Container for audio data and metadata."""

    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float  # Start time of this segment


class AudioDevice:
    """Manages audio device selection and capabilities."""

    def __init__(self: "AudioDevice") -> None:
        """Initialize the audio device interface."""
        self.py_audio = pyaudio.PyAudio()

    def list_devices(self: "AudioDevice") -> list[dict[str, Any]]:
        """List all available audio input devices.

        Returns:
            List of dictionaries containing device information
        """
        devices = []
        for i in range(self.py_audio.get_device_count()):
            try:
                device_info = self.py_audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:  # Only include input devices
                    devices.append(
                        {
                            "index": i,
                            "name": device_info["name"],
                            "channels": device_info["maxInputChannels"],
                            "sample_rate": device_info["defaultSampleRate"],
                        },
                    )
            except (OSError, ValueError) as e:
                logger.warning("Could not get info for device %d: %s", i, e)
        return devices

    def get_default_device(self: "AudioDevice") -> dict[str, Any] | None:
        """Get the default audio input device.

        Returns:
            Dictionary containing device information or None if not available
        """
        try:
            device_info = self.py_audio.get_default_input_device_info()
            return {
                "index": device_info["index"],
                "name": device_info["name"],
                "channels": device_info["maxInputChannels"],
                "sample_rate": device_info["defaultSampleRate"],
            }
        except (OSError, ValueError) as e:
            logger.error("Error getting default device: %s", e)
            return None

    def __del__(self: "AudioDevice") -> None:
        """Clean up PyAudio instance."""
        try:
            self.py_audio.terminate()
        except (OSError, ValueError) as e:
            logger.error("Error terminating PyAudio: %s", e)


class AudioRecorder:
    """Handles real-time audio recording with support for multiple channels."""

    def __init__(self: "AudioRecorder", config: AudioConfig | None = None) -> None:
        """Initialize the audio recorder.

        Args:
            config: Optional audio config. If not provided, loads from default.
        """
        self.config = config or config_manager.load_config().audio
        self.audio_device = AudioDevice()
        self.stream: pyaudio.Stream | None = None
        self.py_audio: pyaudio.PyAudio | None = None
        self.recording = False
        self.audio_queue: Queue[AudioSegment] = Queue()
        self.stop_event = Event()
        self.record_thread: Thread | None = None

    @contextmanager
    def open_stream(self: "AudioRecorder") -> Iterator[None]:
        """Context manager for handling audio stream lifecycle."""
        try:
            self.py_audio = pyaudio.PyAudio()
            self.stream = self.py_audio.open(
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size,
            )
            yield
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.py_audio:
                self.py_audio.terminate()

    def _record_chunk(self: "AudioRecorder") -> np.ndarray | None:
        """Record a single chunk of audio data.

        Returns:
            Numpy array containing audio data or None if recording failed
            :rtype: object
        """
        try:
            if self.stream:
                data = self.stream.read(self.config.chunk_size)
                return np.frombuffer(data, dtype=np.float32)
            return None
        except (OSError, ValueError) as e:
            logger.error("Error recording audio chunk: %s", e)
            return None

    def _recording_worker(self: "AudioRecorder") -> None:
        """Worker thread for continuous recording."""
        with self.open_stream():
            while not self.stop_event.is_set():
                chunk = self._record_chunk()
                if chunk is not None:
                    segment = AudioSegment(
                        data=chunk,
                        sample_rate=self.config.sample_rate,
                        channels=self.config.channels,
                        timestamp=self.audio_queue.qsize()
                        * (self.config.chunk_size / self.config.sample_rate),
                    )
                    self.audio_queue.put(segment)

    def start_recording(self: "AudioRecorder") -> None:
        """Start recording audio in a separate thread."""
        if self.recording:
            logger.warning("Recording already in progress")
            return

        self.recording = True
        self.stop_event.clear()
        self.record_thread = Thread(target=self._recording_worker)
        self.record_thread.start()
        logger.info("Started audio recording")

    def stop_recording(self: "AudioRecorder") -> np.ndarray | None:
        """Stop recording and return the accumulated audio data.

        Returns:
            Numpy array containing the complete recording or None if no data
        """
        if not self.recording:
            logger.warning("No recording in progress")
            return None

        self.stop_event.set()
        if self.record_thread:
            self.record_thread.join()
        self.recording = False

        # Collect all audio segments
        segments = []
        while not self.audio_queue.empty():
            segments.append(self.audio_queue.get())

        if not segments:
            return None

        # Concatenate all segments
        audio_data = np.concatenate([segment.data for segment in segments])
        logger.info("Stopped recording. Captured %d segments", len(segments))
        return audio_data


class AudioLoader:
    """Handles loading audio from various file formats."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @staticmethod
    def load_file(file_path: str | Path) -> tuple[np.ndarray, int]:
        """Load an audio file and return the data and sample rate.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If the file doesn't exist
            SoundFileError: For audio loading errors
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in AudioLoader.SUPPORTED_FORMATS:
            msg = f"Unsupported audio format: {file_path.suffix}"
            raise ValueError(msg)

        if not file_path.exists():
            msg = f"Audio file not found: {file_path}"
            raise FileNotFoundError(msg)

        try:
            audio_data, sample_rate = sf.read(file_path)
            # Ensure float32 type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            logger.info(
                "Loaded audio file: %s (SR: %dHz, Channels: %d)",
                file_path.name,
                sample_rate,
                audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
            )
            return audio_data, sample_rate
        except SoundFileError as e:
            logger.error("Error loading audio file %s: %s", file_path, e)
            raise

    @staticmethod
    def save_wav(
        file_path: str | Path,
        audio_data: np.ndarray,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        """Save audio data to a WAV file.

        Args:
            file_path: Path to save the WAV file
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            channels: Number of audio channels

        Raises:
            OSError: If there are file system related errors
            SoundFileError: If there are audio format related errors
        """
        file_path = Path(file_path)
        try:
            # Ensure the data is float32
            audio_data = audio_data.astype(np.float32)
            sf.write(str(file_path), audio_data, sample_rate, "FLOAT")
            logger.info("Saved audio to: %s", file_path)
        except (OSError, SoundFileError) as e:
            logger.error("Error saving WAV file %s: %s", file_path, e)
            raise
