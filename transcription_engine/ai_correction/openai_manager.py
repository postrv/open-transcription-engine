# File: transcription_engine/ai_correction/openai_manager.py
"""OpenAI integration module for AI-powered transcript correction.

Provides a managed interface to OpenAI's API with retry logic and error handling.
"""

import logging
import os
from typing import TypeVar

import dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for the OpenAIManager class
T = TypeVar("T", bound="OpenAIManager")


class TranscriptCorrection(BaseModel):
    """Response model for transcript corrections."""

    original_text: str
    corrected_text: str
    confidence: float


class OpenAIManager:
    """Manages OpenAI API interactions for transcript correction."""

    CORRECTION_PROMPT = """You are a professional legal transcript editor. 
Please correct any mistakes in this court transcript excerpt, returning only the corrected text. 
Maintain formal language and legal terminology. If no corrections are needed, return the original text.

Original text:
{text}

Corrected text:"""

    def __init__(self: T) -> None:
        """Initialize the OpenAI manager."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "OpenAI API key not found in environment"
            raise ValueError(msg)

        self.client = OpenAI(api_key=api_key)

    def correct_segment(self: T, text: str) -> TranscriptCorrection:
        """Correct a transcript segment using OpenAI.

        Args:
            text: Original transcript text to correct

        Returns:
            TranscriptCorrection containing original and corrected text

        Raises:
            RuntimeError: If API request fails
            ValueError: If input text is invalid
        """
        if not text.strip():
            msg = "Empty text provided"
            raise ValueError(msg)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional legal transcript editor.",
                    },
                    {
                        "role": "user",
                        "content": self.CORRECTION_PROMPT.format(text=text),
                    },
                ],
                temperature=0.3,  # Lower temperature for more conservative corrections
                max_tokens=len(text.split()) * 2,  # Reasonable limit based on input
            )

            corrected_text = response.choices[0].message.content.strip()

            # Calculate a simple confidence score based on edit distance
            from rapidfuzz.distance import Levenshtein

            edit_distance = Levenshtein.distance(text, corrected_text)
            max_length = max(len(text), len(corrected_text))
            confidence = 1.0 - (edit_distance / max_length if max_length > 0 else 0)

            return TranscriptCorrection(
                original_text=text,
                corrected_text=corrected_text,
                confidence=confidence,
            )

        except Exception as e:
            logger.error("Error correcting transcript: %s", e)
            raise RuntimeError(f"Failed to correct transcript: {e}") from e