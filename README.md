# Open Transcription Engine

A local/offline transcription engine focused on generating accurate and privacy-respecting court transcripts.  
This project integrates:
- **Local Whisper Inference (offline)** for transcription
- **Speaker Diarization** for identifying individual speakers
- **Redaction Tools** (both automated and manual)
- **Fuzzy/Phoneme Matching** to catch near-miss sensitive words
- **Timeline Visualization** to review/edit transcripts efficiently

<br/>

## Table of Contents

1. [Background & Goals](#background--goals)  
2. [Architecture Overview](#architecture-overview)  
3. [Project Plan Checklist](#project-plan-checklist)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Next Steps: Code & Features](#next-steps-code--features)  
7. [Contributing](#contributing)  
8. [License](#license)

<br/>

## Background & Goals

Court transcripts in the UK can be very expensive and time-consuming to produce, often requiring manual transcription services. This project aims to build an **open-source transcription and redaction tool** that can:

- **Reduce costs** and speed up the delivery of transcripts to those who need them (victims, lawyers, etc.).
- **Preserve privacy** by allowing both automated redactions (based on a known list of sensitive phrases) and manual redactions (via a user interface).
- **Operate fully offline**, ensuring sensitive audio and text never leave the local machine.
- **Be extensible** to a variety of environments (courtrooms, depositions, legal offices, and more).

<br/>

## Architecture Overview

TODO: Add a high-level diagram of the system architecture, showing the flow of audio input to final transcript output.

<br/>


1. **Audio Input**: Capture multi-channel audio in real-time (or load from existing files).  
2. **Whisper Engine**: Run offline transcription using Whisper models (tiny → large).  
3. **Speaker Identification**: Label speaker segments automatically or per audio channel.  
4. **Redaction**:  
   - Automated approach: Use a defined dictionary of sensitive phrases.  
   - Manual approach: Provide a GUI or step for users to mark text for redaction.  
5. **Fuzzy Matching**: Catch approximate matches (e.g., "Jon" ~ "John") and flag for review.  
6. **Timeline Visualization**: Allow quick inspection and editing of transcripts via a timeline UI.  

<br/>

## Project Plan Checklist

1. **Audio Recorder (Real-Time or File Input)**
   - [ ] Implement multi-channel microphone recording or load from an audio file.  
   - [ ] Ensure robust handling of audio streaming in a courtroom setting.  

2. **Local Whisper Integration**
   - [ ] Add optional GPU support (if available).  
   - [ ] Allow model size selection (tiny, base, small, medium, large).  
   - [ ] Chunk or batch audio input for long recordings.  

3. **Speaker Diarization**
   - [ ] Basic approach: numeric channel assignment if multi-channel audio.  
   - [ ] Advanced approach: incorporate a diarization library (e.g., pyannote) for single-track audio.  

4. **Redaction**
   - [ ] Implement a dictionary-based auto-redaction pipeline.  
   - [ ] Provide a manual redaction workflow in the UI (highlight text, mark it).  
   - [ ] Ensure a “finalization” step that fully censors chosen text.  

5. **Fuzzy/Phoneme Matching**
   - [ ] Integrate a fuzzy library (rapidfuzz, fuzzywuzzy) to detect near-misses.  
   - [ ] Provide a short list of known sensitive references (e.g., partial names, addresses).  
   - [ ] Generate a list of flagged items for manual review.  

6. **Timeline Visualization**
   - [ ] Create a simple web or desktop-based UI to display speaker segments over time.  
   - [ ] Display redaction highlights and flagged terms for user confirmation.  

7. **Output & Security**
   - [ ] Export transcripts (plain text, PDF, or Word).  
   - [ ] Consider encryption or secure storage of raw audio and partial transcripts.  
   - [ ] Add logs or version control to track changes (audit trail).  

8. **Testing & CI/CD**
   - [ ] Write unit tests for each module.  
   - [ ] Set up GitHub Actions or similar CI to run tests automatically.  

9. **Deployment Considerations**
   - [ ] Provide instructions for local installation (Python, dependencies, GPU drivers).  
   - [ ] Offer Docker container or similar packaging for easier on-premises usage.  
   - [ ] Document recommended hardware specs for real-time usage.  

<br/>

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/open-transcription-engine.git
   cd open-transcription-engine
    ```
2. **Activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
     ```
4. **Run the main script**:
    ```bash
    python main.py
     ```

<br/>

## Usage

    Run the main script:

```bash
python main.py
```


Check logs: 
- By default, logs or print statements will be shown in the console.

Configuration:
- Update `transcription_engine/utils/config.py` (or a future `config.yaml`) to customize model size, sensitive word lists, etc.

<br/>

## Next Steps: Code & Features

- Recorder Module: Implement real-time multi-channel audio capture.
- Whisper Integration: Add an actual Whisper import, load a model, and handle transcription.
- Diarization: If using multichannel audio, each channel might map to a distinct speaker. For single-channel, evaluate pyannote.audio.
- UI/Timeline: Build a minimal local web server (e.g., with Flask or FastAPI) to visualize transcripts with a waveform or timeline.
- Security: Investigate ways to encrypt transcripts and manage user access in a real court environment.

<br/>

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements, bug fixes, or new ideas.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request describing your changes

<br/>

## License

MIT License — Feel free to use and adapt this project for non-commercial or commercial purposes, but please give attribution back to this repository.

<br/>