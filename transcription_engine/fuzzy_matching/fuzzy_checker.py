# File: transcription_engine/fuzzy_matching/fuzzy_checker.py
"""
Fuzzy/Phoneme Matching module for detecting approximate matches of sensitive terms.
Uses rapidfuzz for efficient fuzzy string matching and phonetics for name matching.
"""

import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from rapidfuzz import fuzz, process
import jellyfish  # For phonetic matching
from ..utils.config import config_manager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FuzzyMatch:
    """Container for detected fuzzy matches."""
    original_text: str
    matched_term: str
    matched_phrase: str
    confidence: float
    start_pos: int
    end_pos: int
    match_type: str  # 'fuzzy', 'phonetic', or 'partial'


class FuzzyChecker:
    """
    Handles fuzzy matching of sensitive terms with configurable thresholds.
    Supports both fuzzy string matching and phonetic matching for names.
    """

    def __init__(self, config=None):
        """Initialize the fuzzy checker with configuration."""
        self.config = config or config_manager.load_config().redaction
        self.sensitive_terms = self._load_sensitive_terms()
        self.name_terms = {term for term in self.sensitive_terms
                           if len(term.split()) <= 2}  # Likely names
        self.fuzzy_threshold = self.config.fuzzy_threshold
        self.min_length = self.config.min_phrase_length

    def _load_sensitive_terms(self) -> Set[str]:
        """Load sensitive terms from configured file."""
        try:
            path = Path(self.config.sensitive_phrases_file)
            if not path.exists():
                logger.warning(f"Sensitive phrases file not found: {path}")
                return set()

            with open(path, 'r') as f:
                terms = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(terms)} sensitive terms")
            return terms
        except Exception as e:
            logger.error(f"Error loading sensitive terms: {e}")
            return set()

    def _get_phonetic_matches(self, text: str, phrase: str) -> List[FuzzyMatch]:
        """Separate method for phonetic matching of names."""
        matches = []
        phrase_words = phrase.split()

        if len(phrase_words) <= 2:  # Only match potential names
            for term in self.name_terms:
                term_words = term.split()
                if len(phrase_words) == len(term_words):
                    # Compare each word using Soundex
                    if all(jellyfish.soundex(pw) == jellyfish.soundex(tw)
                           for pw, tw in zip(phrase_words, term_words)):
                        matches.append(FuzzyMatch(
                            original_text=text,
                            matched_term=term,
                            matched_phrase=phrase,
                            confidence=0.85,
                            start_pos=text.find(phrase),
                            end_pos=text.find(phrase) + len(phrase),
                            match_type='phonetic'
                        ))
        return matches

    def find_similar_terms(self, transcript_segments: List[Dict]) -> List[FuzzyMatch]:
        """
        Find potential matches for sensitive terms in transcript segments.

        Args:
            transcript_segments: List of transcript segments to check

        Returns:
            List of FuzzyMatch objects for review
        """
        matches = []

        for segment in transcript_segments:
            text = segment['text']
            if not text:
                continue

            # Check each word/phrase in the text
            words = text.split()
            for i in range(len(words)):
                for j in range(i + self.min_length, min(i + 5, len(words) + 1)):
                    phrase = ' '.join(words[i:j])

                    # Skip very short phrases
                    if len(phrase) < self.min_length:
                        continue

                    # First try phonetic matching for names
                    phonetic_matches = self._get_phonetic_matches(text, phrase)
                    if phonetic_matches:
                        matches.extend(phonetic_matches)
                        continue  # Skip fuzzy matching if we found phonetic matches

                    # Then try fuzzy matching
                    fuzzy_matches = process.extract(
                        phrase,
                        self.sensitive_terms,
                        scorer=fuzz.token_sort_ratio,
                        score_cutoff=self.fuzzy_threshold * 80,
                        limit=3
                    )

                    for term, score, _ in fuzzy_matches:
                        matches.append(FuzzyMatch(
                            original_text=text,
                            matched_term=term,
                            matched_phrase=phrase,
                            confidence=score / 100.0,
                            start_pos=text.find(phrase),
                            end_pos=text.find(phrase) + len(phrase),
                            match_type='fuzzy'
                        ))

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.matched_term, match.start_pos, match.end_pos)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        logger.info(f"Found {len(unique_matches)} potential sensitive term matches")
        return unique_matches

    def save_matches(self, matches: List[FuzzyMatch], output_path: Path):
        """Save matches to a JSON file for review."""
        try:
            matches_dict = [
                {
                    'original_text': m.original_text,
                    'matched_term': m.matched_term,
                    'matched_phrase': m.matched_phrase,
                    'confidence': float(m.confidence),
                    'start_pos': m.start_pos,
                    'end_pos': m.end_pos,
                    'match_type': m.match_type
                }
                for m in matches
            ]

            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to string first
            json_str = json.dumps(matches_dict, indent=2)

            # Write content
            with open(output_path, 'w') as f:
                f.write(json_str)
                f.flush()  # Ensure content is written

            logger.info(f"Saved {len(matches)} matches to {output_path}")

        except Exception as e:
            logger.error(f"Error saving matches: {e}")
            raise