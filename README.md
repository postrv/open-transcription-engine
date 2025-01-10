# Open Transcription Engine

A local/offline transcription engine focused on generating accurate and privacy-respecting court transcripts.

## Project Status: Alpha Development

### ✅ Completed Core Components
- Audio recording and processing pipeline with PyAudio
- Whisper integration with MPS/CUDA/GPU support
- Speaker diarization framework (multi-channel + ML-based)
- Fuzzy matching system for redaction
- Redaction framework (auto + manual)
- Timeline UI components (FastAPI + React)
- Comprehensive test suite
- Modern Python tooling setup (ruff, mypy, pre-commit)

### 🚧 Current Issues
1. **Missing Dependencies**
   - pyannote.audio needs to be installed for speaker diarization
   - Currently affects one test suite (test_diarizer.py)

2. **Code Quality**
   - Type annotations missing in test files
   - Docstring formatting issues
   - Some test assertions need safety improvements
   - Line length violations in tests

### 🎯 Immediate Next Steps
1. **Fix Dependencies**
   ```bash
   # Add to environment.yml
   pip:
     - pyannote.audio
   ```

2. **Code Quality**
   - Add proper type annotations to test files
   - Fix docstring formatting
   - Update assert statements to use pytest's assertion helpers
   - Address line length issues

3. **Documentation**
   - Add docstrings to __init__.py files
   - Complete API documentation
   - Add architecture diagrams
   - Create user guides

## Features

### Audio Processing
- Multi-channel audio capture
- File loading for various formats
- Real-time streaming support

### Transcription
- Local Whisper inference
- Support for multiple GPU types (MPS/CUDA)
- Configurable model sizes

### Speaker Identification
- Channel-based diarization
- ML-based speaker separation
- Integration with pyannote.audio

### Privacy & Security
- Fully offline operation
- Automated redaction system
- Manual redaction interface
- Fuzzy matching for sensitive terms

### User Interface
- Web-based timeline view
- Real-time editing capabilities
- Redaction zone management

## Installation

1. **Prerequisites**
   - Python 3.12+
   - Conda/Miniforge
   - Git

2. **Setup**
   ```bash
   # Clone repository
   git clone https://github.com/YourUsername/open-transcription-engine.git
   cd open-transcription-engine

   # Create environment
   conda env create -f environment.yml
   conda activate whisper

   # Install dev tools
   pre-commit install
   ```

3. **Configure**
   - Edit `config/default.yml` for your setup
   - Default configuration uses MPS on Apple Silicon

## Development

### Code Quality Tools
```bash
# Run all checks
pre-commit run --all-files

# Individual tools
ruff check .
ruff format .
mypy .
pytest
```

### Testing
```bash
# Run all tests
pytest

# With coverage
pytest --cov=transcription_engine

# Single test file
pytest transcription_engine/tests/test_recorder.py
```

## Usage

Currently in development. Basic usage:

```python
from transcription_engine.audio_input import recorder
from transcription_engine.whisper_engine import transcriber

# Initialize components
audio_recorder = recorder.AudioRecorder()
whisper_manager = transcriber.WhisperManager()

# Record audio
audio_recorder.start_recording()
# ... wait for recording ...
audio_data = audio_recorder.stop_recording()

# Transcribe
segments = whisper_manager.transcribe(audio_data, sample_rate=16000)
```

## Contributing

1. Fork the repository
2. Install development tools: `pre-commit install`
3. Make changes
4. Ensure all tests pass: `pytest`
5. Submit PR

## License

MIT License - See LICENSE file for details
