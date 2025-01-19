# Issue Tracker

## Core UI Improvements

### Timeline & Audio Visualization ✅
- [x] Implement vertical list layout
- [x] Integrate wavesurfer.js for audio visualization
  - [x] Add waveform display synchronized with transcript
  - [x] Implement click-to-seek functionality
  - [x] Add playback controls (play/pause/stop)
  - [x] Display current time and segment highlighting

### File Upload and Processing
- [x] Connect upload endpoint to transcription engine
- [x] Add upload validation
  - [x] File size limits
  - [x] Audio format validation
- [ ] Improve real-time status updates
  - [ ] Add better progress tracking
  - [ ] Enhance websocket reliability
- [ ] Improve error handling
  - [ ] User-friendly error messages
  - [ ] Retry mechanism for failed uploads

### Speaker Management (High Priority)
- [ ] Fix accuracy issues in diarization
  - [ ] Improve speaker separation algorithm
  - [ ] Better integration with pyannote.audio
  - [ ] Add validation for speaker detection
- [ ] Implement more robust confidence scoring
  - [ ] Add ground truth validation
  - [ ] Improve confidence calculation
  - [ ] Add confidence visualization

### Editing & Navigation (In Progress)
- [x] Basic in-place editing
- [ ] Enhance editing features
  - [ ] Add auto-save functionality
  - [ ] Implement keyboard shortcuts
  - [ ] Add undo/redo capability
- [ ] Add search and filter functionality

### Redaction System
- [x] Basic auto-redaction framework
- [x] Manual redaction zones
- [ ] Enhance redaction UI
  - [ ] Better text selection interface
  - [ ] Visual indicators for redacted zones
  - [ ] Add redaction review/edit workflow
- [ ] Add export options for redacted content

## Performance Optimization
- [x] Basic MPS support for M1 Macs
- [ ] Optimize memory usage
- [ ] Improve processing speed
- [ ] Add progress monitoring

## Recent Improvements
- Upgraded to newer Whisper model versions
- Implemented speaker diarization with confidence scoring
- Added waveform visualization with wavesurfer.js
- Improved transcription accuracy
- Added basic confidence level display
- Multiple speaker detection working

## Current Limitations

### Speaker Diarization
- Current implementation has accuracy issues
- May not correctly distinguish between speakers
- Channel separation needs improvement
- Integration with pyannote.audio needs optimization

### Confidence Scoring
- Confidence levels may not accurately reflect transcription quality
- Further calibration needed
- Validation against ground truth required

## Priority Items

### Diarization Improvements
- [ ] Investigate alternative diarization approaches
- [ ] Add speaker validation tools
- [ ] Implement manual speaker correction
- [ ] Add batch processing for diarization

### Confidence Scoring Enhancement
- [ ] Develop better confidence metrics
- [ ] Add visual confidence indicators
- [ ] Implement confidence thresholds
- [ ] Add confidence filtering options
