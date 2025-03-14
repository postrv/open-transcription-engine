# Open Transcription Engine

A local/offline transcription engine focused on generating accurate and privacy-respecting court transcripts.

<img src="output.gif">

### Project Status: Beta Development

## ✅ Working Features

- Audio file processing pipeline with support for MP3, WAV, FLAC, OGG, M4A
- Whisper integration with MPS/CUDA support
- Basic transcription functionality working end-to-end
- Timeline UI with audio upload
- FastAPI backend integration
- Wavesurfer.js integration for audio visualization
- JSON export functionality
- Basic redaction framework

## 🚧 Areas Needing Improvement

- Speaker diarization accuracy needs enhancement
- Confidence scoring reliability needs work
- Real-time progress updates during transcription
- Redaction zone UI/UX improvements
- Additional export formats (PDF, SRT, plain text)

## Features

### Audio Processing
- Multi-channel audio capture
- Support for common audio formats (WAV, MP3, FLAC, OGG, M4A)
- Real-time streaming support
- Efficient batch processing (~20min for 3hr files on M1)

### Transcription
- Local Whisper inference with latest models
- GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA)
- Configurable model sizes (tiny to large-v3)
- Confidence scoring for each segment

### Speaker Identification
- Channel-based diarization for multi-track audio
- ML-based speaker separation with pyannote.audio
- Speaker confidence metrics
- Overlap detection

### Privacy & Security
- Fully offline operation
- Automated redaction system
- Manual redaction interface
- Fuzzy matching for sensitive terms
- Configurable redaction rules

### User Interface
- Web-based timeline view
- Audio waveform visualization
- Real-time editing capabilities
- Redaction zone management

## Installation

1. **Prerequisites**
   - Python 3.12+
   - Conda/Miniforge
   - Git
   - PortAudio (for audio capture)
   - Node.js & npm (for UI development)

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
   - Set up `.env` file for environment variables in the root directory

## Development

### Running the Application

1. **Start the Backend**
```bash
# From project root
python -m uvicorn transcription_engine.timeline_visualization.timeline_ui:app --reload --port 8000
```

2. **Start the Frontend**
```bash
# In another terminal
cd transcription_engine/static
npm install
npm run dev
```

### Performance Notes

#### GPU Configuration

- MPS (Metal Performance Shaders) used by default on Apple Silicon
- CUDA support available for NVIDIA GPUs
- Automatic CPU fallback if GPU unavailable

#### Memory Usage

- Default batch_size=8 optimized for M1/M2
- Adjust based on available memory:
  - tiny/base: 1GB minimum
  - small: 2GB minimum
  - medium: 5GB minimum
  - large: 10GB minimum

#### Processing Speed

- ~20 minutes for 3hr files on M1 (MPS)
- Use --model tiny for rapid development
- Parallel processing available for batch files

### Code Quality

```bash
# Run all checks
pre-commit run --all-files

# Individual tools
ruff check .
ruff format .
mypy .
pytest

# Coverage
pytest --cov=transcription_engine
```

## API Usage

### Basic Python Usage
```python
from transcription_engine.audio_input.recorder import AudioLoader
from transcription_engine.whisper_engine.transcriber import WhisperManager
from transcription_engine.redaction.redactor import TranscriptRedactor

# Load and transcribe
audio_data, sample_rate = AudioLoader.load_file("input.mp3")
whisper_manager = WhisperManager()
whisper_manager.load_model()
segments = whisper_manager.transcribe(audio_data, sample_rate)

# Apply redaction
redactor = TranscriptRedactor()
redacted_segments, matches = redactor.auto_redact(segments)
```

### HTTP API Endpoints

#### Transcript Management
- `POST /api/transcript/load` - Load transcript data
- `GET /api/transcript` - Retrieve current transcript
- `GET /api/transcript/{job_id}` - Get specific job transcript

#### Processing
- `POST /api/upload-audio` - Upload audio file for processing
- `WS /ws/jobs/{job_id}` - WebSocket for processing updates

#### Redaction
- `POST /api/redaction` - Add redaction zone

## Common Issues

### GPU/Hardware
- System falls back to CPU if MPS/CUDA unavailable
- Monitor memory usage during processing
- Reduce batch size if OOM errors occur

### Audio Files
- Verify file format compatibility
- Check file permissions
- Use ffmpeg for format conversion if needed

### Processing
- Large files may need model size adjustment
- Speaker diarization accuracy varies
- Confidence scores need validation

## Development Testing

```python
# Load test data
with open("transcription_engine/tests/data/test.json") as f:
    test_data = json.load(f)

# Test phrases
with open("transcription_engine/tests/data/test_phrases.txt") as f:
    test_phrases = f.read().splitlines()
```

## Contributing

1. Fork repository
2. Install tools: `pre-commit install`
3. Make changes
4. Run tests: `pytest`
5. Submit PR

## License

MIT License - See LICENSE file for details
