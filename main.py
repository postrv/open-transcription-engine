# File: main.py
"""Main entry point for the Courtroom Transcription & Redaction System."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from transcription_engine.audio_input.recorder import AudioLoader
from transcription_engine.redaction.redactor import TranscriptRedactor
from transcription_engine.speaker_id.diarizer import DiarizationManager
from transcription_engine.utils.config import config_manager
from transcription_engine.whisper_engine.transcriber import WhisperManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def process_audio(
    input_path: Path,
    output_dir: Path,
    model_size: str,
    use_diarization: bool = True,
    hf_token: str | None = None,
) -> None:
    """Process an audio file through the transcription pipeline.

    Args:
        input_path: Path to input audio file
        output_dir: Directory for output files
        model_size: Whisper model size to use
        use_diarization: Whether to use speaker diarization
        hf_token: HuggingFace token for pyannote.audio (if using diarization)
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Load audio file
            task_id = progress.add_task(
                f"Loading audio file: {input_path}",
                total=None,
            )
            audio_data, sample_rate = AudioLoader.load_file(input_path)
            progress.update(task_id, completed=True)

            # Initialize Whisper
            task_id = progress.add_task(
                f"Loading Whisper model: {model_size}",
                total=None,
            )
            whisper_manager = WhisperManager()
            whisper_manager.config.model_size = model_size  # Override model size
            if not whisper_manager.load_model():
                msg = "Failed to load Whisper model"
                raise RuntimeError(msg)
            progress.update(task_id, completed=True)

            # Transcribe
            task_id = progress.add_task(
                "Transcribing audio",
                total=None,
            )
            segments = whisper_manager.transcribe(
                audio_data,
                sample_rate,
                progress_callback=lambda x: progress.update(
                    task_id,
                    description=f"Transcribing audio: {x:.1f}%",
                ),
            )
            progress.update(task_id, completed=True)

            # Speaker diarization
            if use_diarization:
                task_id = progress.add_task(
                    "Running speaker diarization",
                    total=None,
                )
                config = config_manager.load_config()
                if hf_token:  # Override token from CLI if provided
                    config.diarization.auth_token = hf_token
                diarizer = DiarizationManager(config.diarization)
                diarization_segments = diarizer.process_singlechannel(
                    audio_data,
                    sample_rate,
                )
                segments = diarizer.assign_speaker_ids(segments, diarization_segments)
                progress.update(task_id, completed=True)

            # Initialize redactor
            task_id = progress.add_task(
                "Running automatic redaction",
                total=None,
            )
            redactor = TranscriptRedactor()
            redacted_segments, matches = redactor.auto_redact(segments)
            progress.update(task_id, completed=True)

            # Save outputs
            task_id = progress.add_task("Saving outputs", total=None)
            output_path = output_dir / f"{input_path.stem}_redacted.json"
            redactor.save_redactions(redacted_segments, output_path)

            matches_path = output_dir / f"{input_path.stem}_matches.json"
            redactor.fuzzy_checker.save_matches(matches, matches_path)
            progress.update(task_id, completed=True)

        console.print(f"\n✨ Processing complete! Output saved to: {output_path}")

    except Exception as e:
        logger.error("Error processing file: %s", e, exc_info=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (optional)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    valid_models = {
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    }

    parser = argparse.ArgumentParser(
        description="Process court recordings with privacy-preserving transcription",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file or directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: ./output)",
        default=Path("output"),
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for pyannote.audio (required for diarization)",
    )
    parser.add_argument(
        "--model",
        choices=valid_models,
        default="large",
        help="Whisper model size to use",
    )

    args = parser.parse_args(argv)

    try:
        # Load configuration
        config = config_manager.load_config()

        if not config.output_dir.exists():
            config.output_dir.mkdir(parents=True)

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Process single file or directory
        if args.input.is_file():
            process_audio(
                args.input,
                args.output_dir,
                args.model,
                not args.no_diarization,
                args.hf_token,
            )
        elif args.input.is_dir():
            for file_path in args.input.glob("*"):
                if file_path.suffix.lower() in AudioLoader.SUPPORTED_FORMATS:
                    process_audio(
                        file_path,
                        args.output_dir,
                        args.model,
                        not args.no_diarization,
                        args.hf_token,
                    )
        else:
            logger.error("Input path does not exist: %s", args.input)
            return 1

        return 0

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
