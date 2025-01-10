# Open Transcription Engine

A local/offline transcription engine focused on generating accurate and privacy-respecting court transcripts.
This project integrates:
- **Local Whisper Inference (offline)** for transcription
- **Speaker Diarization** for identifying individual speakers
- **Redaction Tools** (both automated and manual)
- **Fuzzy/Phoneme Matching** to catch near-miss sensitive words
- **Timeline Visualization** to review/edit transcripts efficiently

## Table of Contents

1. [Background & Goals](#background--goals)
2. [Architecture Overview](#architecture-overview)
3. [Project Status](#project-status)
4. [Installation](#installation)
5. [Development Setup](#development-setup)
6. [Usage](#usage)
7. [Next Steps](#next-steps)
8. [Contributing](#contributing)
9. [License](#license)

## Background & Goals

Court transcripts in the UK can be very expensive and time-consuming to produce, often requiring manual transcription services. This project aims to build an **open-source transcription and redaction tool** that can:

- **Reduce costs** and speed up the delivery of transcripts to those who need them (victims, lawyers, etc.).
- **Preserve privacy** by allowing both automated redactions (based on a known list of sensitive phrases) and manual redactions (via a user interface).
- **Operate fully offline**, ensuring sensitive audio and text never leave the local machine.
- **Be extensible** to a variety of environments (courtrooms, depositions, legal offices, and more).

## Architecture Overview

The system processes audio in several stages:

1. **Audio Input**: Multi-channel audio capture with real-time streaming and file input support
2. **Whisper Engine**: Offline transcription using Whisper models with MPS/CUDA/CPU support
3. **Speaker Identification**:
   - Channel-based diarization for multi-channel audio
   - ML-based diarization (optional pyannote.audio) for single-channel
4. **Redaction Pipeline**:
   - Automated redaction using configurable sensitive phrase lists
   - Fuzzy matching to catch approximate matches
   - Manual redaction through web UI
5. **Timeline UI**: FastAPI + React-based interface for transcript review and editing

## Project Status

✅ Completed:
- Base architecture and module structure
- Audio recording and processing pipeline
- Whisper integration with GPU support
- Basic speaker diarization
- Fuzzy matching system
- Redaction framework
- Timeline UI components
- Comprehensive test suite

🚧 In Progress:
- Timeline UI server implementation
- React component integration
- Manual redaction workflow
- Documentation improvements

## Installation

1. **Prerequisites**:
   - Python 3.10+
   - Git
   - [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/open-transcription-engine.git
   cd open-transcription-engine
   ```

3. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate whisper
   ```

4. **Install development tools**:
   ```bash
   pre-commit install
   ```

## Development Setup

This project uses several development tools to ensure code quality:

- **Ruff**: For linting and formatting (replaces Black/isort/flake8)
- **MyPy**: For static type checking
- **Pre-commit**: For automated checks before commits
- **Pytest**: For testing
- **Coverage**: For test coverage reporting

To run code quality checks:
```bash
# Run Ruff formatting and linting
ruff check .
ruff format .

# Run type checking
mypy .

# Run tests with coverage
pytest --cov=transcription_engine
```

## Usage

1. **Configure the system**:
   Edit `config/default.yml` to set:
   - Whisper model size
   - Audio input preferences
   - Sensitive phrase lists
   - GPU/CPU preferences

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Access the Timeline UI**:
   Open `http://localhost:8000` in your browser to:
   - Review transcripts
   - Apply manual redactions
   - Export final versions

## Next Steps

Priority development areas:

1. **UI/Timeline**:
   - Complete FastAPI server implementation
   - Add waveform visualization
   - Implement real-time updates

2. **Security**:
   - Add audio encryption
   - Implement secure storage
   - Add audit logging

3. **Deployment**:
   - Create Docker container
   - Add deployment guides
   - Document hardware requirements

4. **Documentation**:
   - Add API documentation
   - Create user guides
   - Add architecture diagrams

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Install development tools: `pre-commit install`
4. Make your changes
5. Ensure all tests pass: `pytest`
6. Open a Pull Request

## License

MIT License — Feel free to use and adapt this project for non-commercial or commercial purposes, but please give attribution back to this repository.
