# File: transcription_engine/tests/test_fuzzy_checker.py
"""Tests for the fuzzy matching functionality."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ..fuzzy_matching.fuzzy_checker import FuzzyChecker, FuzzyMatch
from ..utils.config import RedactionConfig


@pytest.fixture(scope="function")  # type: ignore[misc]
def mock_config() -> RedactionConfig:
    """Provide test configuration.

    Returns:
        RedactionConfig: Test configuration instance
    """
    return RedactionConfig(
        sensitive_phrases_file=Path("test_phrases.txt"),
        redaction_char="*",
        min_phrase_length=2,
        fuzzy_threshold=0.85,
    )


@pytest.fixture(scope="function")  # type: ignore[misc]
def sample_sensitive_terms() -> str:
    """Provide sample sensitive terms.

    Returns:
        str: Sample sensitive terms for testing
    """
    return """John Smith
Robert Jones
123 Main Street
London SW1
+44 7700 900123"""


class TestFuzzyChecker:
    """Test suite for fuzzy matching functionality."""

    def test_initialization(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test FuzzyChecker initialization.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_open(read_data=sample_sensitive_terms)),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            pytest.assume(checker.fuzzy_threshold == 0.85)
            pytest.assume(checker.min_length == 2)
            pytest.assume(len(checker.sensitive_terms) == 5)
            pytest.assume("John Smith" in checker.name_terms)

    def test_fuzzy_matching(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test fuzzy string matching.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_open(read_data=sample_sensitive_terms)),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            segments = [{"text": "Jon Smyth was present at the hearing"}]
            matches = checker.find_similar_terms(segments)

            pytest.assume(len(matches) >= 1)
            pytest.assume(any(m.matched_term == "John Smith" for m in matches))
            pytest.assume(all(isinstance(m, FuzzyMatch) for m in matches))

    def test_phonetic_matching(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test phonetic matching for names.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_open(read_data=sample_sensitive_terms)),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            segments = [{"text": "Robbert Joanes attended the meeting"}]
            matches = checker.find_similar_terms(segments)

            pytest.assume(len(matches) >= 1)
            pytest.assume(any(m.matched_term == "Robert Jones" for m in matches))
            pytest.assume(any(m.match_type == "phonetic" for m in matches))

    def test_address_matching(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test matching of address patterns.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_open(read_data=sample_sensitive_terms)),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            segments = [{"text": "They lived at 123 Maine Street"}]
            matches = checker.find_similar_terms(segments)

            pytest.assume(len(matches) >= 1)
            pytest.assume(any("123 Main Street" in m.matched_term for m in matches))

    def test_phone_number_matching(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test matching of phone number patterns.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_open(read_data=sample_sensitive_terms)),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            segments = [{"text": "Contact number is +44 7700 900 123"}]
            matches = checker.find_similar_terms(segments)

            pytest.assume(len(matches) >= 1)
            pytest.assume(any("+44 7700 900123" in m.matched_term for m in matches))

    def test_save_matches(
        self: "TestFuzzyChecker",
        mock_config: RedactionConfig,
        sample_sensitive_terms: str,
    ) -> None:
        """Test saving matches to JSON.

        Args:
            mock_config: Test configuration fixture
            sample_sensitive_terms: Sample terms fixture
        """
        mock_file = mock_open(read_data=sample_sensitive_terms)
        output_mock = mock_open()
        mock_file.side_effect = [
            mock_open(read_data=sample_sensitive_terms)(),
            output_mock(),
        ]

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("builtins.open", mock_file),
        ):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)
            segments = [{"text": "Jon Smyth and Robert Joanes were present"}]
            matches = checker.find_similar_terms(segments)

            output_path = Path("test_output.json")
            checker.save_matches(matches, output_path)

            output_mock().write.assert_called()
            write_call_args = output_mock().write.call_args[0][0]
            pytest.assume(len(write_call_args) > 0)

            saved_data = json.loads(write_call_args)
            pytest.assume(isinstance(saved_data, list))
            pytest.assume(len(saved_data) > 0)
            pytest.assume("matched_term" in saved_data[0])

        def test_error_handling(
            self: "TestFuzzyChecker",
            mock_config: RedactionConfig,
        ) -> None:
            """Test error handling for missing files and invalid input.

            Args:
                self: Instance of the test class
                mock_config: Test configuration fixture
            """
            checker = FuzzyChecker(mock_config)
            pytest.assume(len(checker.sensitive_terms) == 0)

            matches = checker.find_similar_terms([])
            pytest.assume(len(matches) == 0)

            matches = checker.find_similar_terms([{"text": ""}])
            pytest.assume(len(matches) == 0)


if __name__ == "__main__":
    pytest.main([__file__])
