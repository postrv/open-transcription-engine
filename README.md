# Open Transcription Engine

A local/offline transcription engine focused on generating accurate and privacy-respecting court transcripts.

### Project Status: Beta Development


## ✅ Completed Core Components

- Audio recording and processing pipeline with PyAudio
- Whisper integration with MPS/CUDA/GPU support (3hr+ audio in ~20min)
- Speaker diarization framework (multi-channel + ML-based)
- Fuzzy matching system for redaction
- Redaction framework (auto + manual)
- Timeline UI components (FastAPI + React)
- Comprehensive test suite
- Modern Python tooling setup (ruff, mypy, pre-commit)

## 🚧 Current Issues

### Frontend Improvements Needed

- Add waveform visualization using wavesurfer.js
- Implement keyboard shortcuts for timeline navigation
- Add real-time progress updates
- Improve redaction zone UI/UX
- Add export options (PDF, SRT, plain text)


### Code Quality

- Type annotations missing in test files
- Docstring formatting issues
- Some test assertions need safety improvements
- Line length violations in tests


### Documentation & Usage

- Missing comprehensive documentation for court staff
- No clear deployment guide
- Security considerations not fully documented



## 🎯 Immediate Next Steps

### Frontend Enhancement

- Implement audio waveform visualization
- Add keyboard controls for timeline
- Improve speaker labeling UI
- Add auto-save functionality
- Implement undo/redo for redactions


### Code Quality

- Add proper type annotations to test files
- Fix docstring formatting
- Update assert statements to use pytest's assertion helpers
- Address line length issues


### Documentation

- Add docstrings to init.py files
- Complete API documentation
- Add architecture diagrams
- Create user guides specific to court settings


### Security & Privacy

- Implement HuggingFace token management for pyannote
- Add data retention policies
- Document compliance with UK court requirements
- Add secure audio file handling


### Performance Optimization

- Fine-tune MPS memory utilization
- Add parallel processing for multi-file batches
- Implement caching for repeated phrases
- Optimize speaker diarization for long files

## Features

### Audio Processing
- Multi-channel audio capture
- File loading for various formats (WAV, MP3, FLAC, OGG, M4A)
- Real-time streaming support

### Transcription
- Local Whisper inference
- Support for multiple GPU types (MPS/CUDA)
- Configurable model sizes (tiny to large)

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
   - PortAudio (for audio capture)

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
   - Copy `config/default.yml.example` to `config/default.yml`
   - Configure GPU settings (defaults to MPS on Apple Silicon)
   - Set up sensitive phrases file
   - Configure audio settings

## Development

### Performance Notes

#### Processing Times

- ~20 minutes for 3hr+ files (MPS/M1)
- Use --model tiny for rapid development
- Adjust batch_size based on available memory


#### Memory Optimization

- Monitor Activity Monitor during processing
- Default batch_size=8 works well on M1
- Reduce for older hardware if needed


#### GPU Utilization

- MPS performs well on M1/M2
- CUDA support should work on NVIDIA but is not yet tested
- Automatic CPU fallback if needed

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

# Testing Guide - Open Transcription Engine

## Setup

1. **Environment Setup**
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate whisper

# Install pre-commit hooks
pre-commit install
```

2. **Configuration**
```bash
# Create config directory if it doesn't exist
mkdir -p config

# Copy the default configuration
cp config/default.yml config/default.yml

# Create sensitive phrases file
echo "John Smith
Jane Doe
+44 7700 900123
123 Main Street
London SW1" > config/sensitive_phrases.txt
```

## Running the Application

1. **Start the FastAPI server**
```bash
# From the project root
python -m uvicorn transcription_engine.timeline_visualization.timeline_ui:app --reload --port 8000
```

2. **Build and watch frontend (in a new terminal)**
```bash
cd transcription_engine/static
npm install
npm run dev
```

## Available Endpoints

### 1. Timeline UI
- Web Interface: `http://localhost:8000/`
- Provides visual interface for transcript review and redaction

### 2. API Endpoints

#### Transcript Management
- `POST /api/transcript/load`
  ```bash
  curl -X POST http://localhost:8000/api/transcript/load \
    -H "Content-Type: application/json" \
    -d @test.json
  ```

- `GET /api/transcript`
  ```bash
  curl http://localhost:8000/api/transcript
  ```

#### Redaction
- `POST /api/redaction`
  ```bash
  curl -X POST http://localhost:8000/api/redaction \
    -H "Content-Type: application/json" \
    -d '{
      "start_time": 0.5,
      "end_time": 1.5,
      "text": "sensitive information",
      "reason": "Personal information"
    }'
  ```

## Testing Your MP3 File

1. **Using Python API**
```python
from pathlib import Path
from transcription_engine.audio_input.recorder import AudioLoader
from transcription_engine.whisper_engine.transcriber import WhisperManager
from transcription_engine.redaction.redactor import TranscriptRedactor

# Load audio file
audio_data, sample_rate = AudioLoader.load_file("path_to_your_court_trial.mp3")

# Initialize Whisper (uses MPS on M1 Mac by default)
whisper_manager = WhisperManager()
whisper_manager.load_model()

# Transcribe
segments = whisper_manager.transcribe(audio_data, sample_rate)

# Initialize redactor
redactor = TranscriptRedactor()

# Auto-redact sensitive information
redacted_segments, matches = redactor.auto_redact(segments)

# Save results
import json
with open("transcript_output.json", "w") as f:
    json.dump({
        "segments": redacted_segments,
        "matches": [vars(m) for m in matches]
    }, f, indent=2)
```

2. **Using Timeline UI**
```python
# Start server then run:
import requests
import json

# Load transcript
with open("transcript_output.json") as f:
    transcript = json.load(f)

# Send to API
response = requests.post(
    "http://localhost:8000/api/transcript/load",
    json=transcript["segments"]
)
print(response.json())

# Now visit http://localhost:8000 to view and edit
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest transcription_engine/tests/test_recorder.py

# Run with coverage
pytest --cov=transcription_engine

# Run with detailed output
pytest -v
```

## Common Issues

1. **MPS/GPU Issues**
- If you encounter MPS errors, the system will fallback to CPU
- Check GPU status:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

2. **Audio File Issues**
- Supported formats: WAV, MP3, FLAC, OGG, M4A
- For unsupported formats, convert using ffmpeg:
```bash
ffmpeg -i input.xxx output.mp3
```

3. **Memory Issues**
- Whisper models have different memory requirements:
  - tiny: 1GB
  - base: 1GB
  - small: 2GB
  - medium: 5GB
  - large: 10GB
- Adjust model size in config/default.yml if needed

## Development Testing

For development, you can use the test data provided:
```python
# Load test transcript
with open("transcription_engine/tests/data/test.json") as f:
    test_data = json.load(f)

# Use test phrases
with open("transcription_engine/tests/data/test_phrases.txt") as f:
    test_phrases = f.read().splitlines()
```

## Contributing

1. Fork the repository
2. Install development tools: `pre-commit install`
3. Make changes
4. Ensure all tests pass: `pytest`
5. Submit PR

## License

MIT License - See LICENSE file for details
