# File: transcription_engine/audio_input/recorder.py
"""
Audio Recorder module for the Open Transcription Engine.
Handles both real-time audio capture and file loading with support for
multi-channel audio commonly found in courtroom settings.
"""

import wave
import pyaudio
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Event
from contextlib import contextmanager

from ..utils.config import config_manager, AudioConfig

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

    def __init__(self):
        self.py_audio = pyaudio.PyAudio()

    def list_devices(self) -> list[dict]:
        """List all available audio input devices."""
        devices = []
        for i in range(self.py_audio.get_device_count()):
            try:
                device_info = self.py_audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only include input devices
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
            except Exception as e:
                logger.warning(f"Could not get info for device {i}: {e}")
        return devices

    def get_default_device(self) -> dict:
        """Get the default audio input device."""
        try:
            device_info = self.py_audio.get_default_input_device_info()
            return {
                'index': device_info['index'],
                'name': device_info['name'],
                'channels': device_info['maxInputChannels'],
                'sample_rate': device_info['defaultSampleRate']
            }
        except Exception as e:
            logger.error(f"Error getting default device: {e}")
            return None

    def __del__(self):
        """Clean up PyAudio instance."""
        try:
            self.py_audio.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")


class AudioRecorder:
    """Handles real-time audio recording with support for multiple channels."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or config_manager.load_config().audio
        self.audio_device = AudioDevice()
        self.stream = None
        self.py_audio = None
        self.recording = False
        self.audio_queue = Queue()
        self.stop_event = Event()

    @contextmanager
    def open_stream(self):
        """Context manager for handling audio stream lifecycle."""
        try:
            self.py_audio = pyaudio.PyAudio()
            self.stream = self.py_audio.open(
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size
            )
            yield
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.py_audio:
                self.py_audio.terminate()

    def _record_chunk(self):
        """Record a single chunk of audio data."""
        try:
            data = self.stream.read(self.config.chunk_size)
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error recording audio chunk: {e}")
            return None

    def _recording_worker(self):
        """Worker thread for continuous recording."""
        with self.open_stream():
            while not self.stop_event.is_set():
                chunk = self._record_chunk()
                if chunk is not None:
                    self.audio_queue.put(AudioSegment(
                        data=chunk,
                        sample_rate=self.config.sample_rate,
                        channels=self.config.channels,
                        timestamp=self.audio_queue.qsize() * (self.config.chunk_size / self.config.sample_rate)
                    ))

    def start_recording(self):
        """Start recording audio in a separate thread."""
        if self.recording:
            logger.warning("Recording already in progress")
            return

        self.recording = True
        self.stop_event.clear()
        self.record_thread = Thread(target=self._recording_worker)
        self.record_thread.start()
        logger.info("Started audio recording")

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the accumulated audio data."""
        if not self.recording:
            logger.warning("No recording in progress")
            return None

        self.stop_event.set()
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
        logger.info(f"Stopped recording. Captured {len(segments)} segments")
        return audio_data



class AudioLoader:
    """Handles loading audio from various file formats."""

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the data and sample rate.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        # 1. Check extension first
        if file_path.suffix.lower() not in AudioLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")

        # 2. Then check existence
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            audio_data, sample_rate = sf.read(file_path)
            # Ensure float32 type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            logger.info(f"Loaded audio file: {file_path.name} "
                       f"(SR: {sample_rate}Hz, Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1})")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    @staticmethod
    def save_wav(file_path: Union[str, Path], audio_data: np.ndarray,
                 sample_rate: int, channels: int = 1):
        """Save audio data to a WAV file."""
        file_path = Path(file_path)
        try:
            # Ensure the data is float32
            audio_data = audio_data.astype(np.float32)
            sf.write(str(file_path), audio_data, sample_rate, 'FLOAT')
            logger.info(f"Saved audio to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving WAV file {file_path}: {e}")
            raise