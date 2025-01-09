# File: transcription_engine/tests/test_fuzzy_checker.py
"""
Unit tests for the fuzzy matching functionality.
"""
import json

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, mock_open
from ..fuzzy_matching.fuzzy_checker import FuzzyChecker, FuzzyMatch
from ..utils.config import RedactionConfig


@pytest.fixture
def mock_config():
    """Fixture providing test configuration."""
    return RedactionConfig(
        sensitive_phrases_file='test_phrases.txt',
        redaction_char='*',
        min_phrase_length=2,
        fuzzy_threshold=0.85
    )


@pytest.fixture
def sample_sensitive_terms():
    """Fixture providing sample sensitive terms."""
    return """John Smith
Robert Jones
123 Main Street
London SW1
+44 7700 900123"""


class TestFuzzyChecker:
    """Test suite for fuzzy matching functionality."""

    def test_initialization(self, mock_config, sample_sensitive_terms):
        """Test FuzzyChecker initialization."""
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_open(read_data=sample_sensitive_terms)):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)
            assert checker.fuzzy_threshold == 0.85
            assert checker.min_length == 2
            assert len(checker.sensitive_terms) == 5
            assert 'John Smith' in checker.name_terms

    def test_fuzzy_matching(self, mock_config, sample_sensitive_terms):
        """Test fuzzy string matching."""
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_open(read_data=sample_sensitive_terms)):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            # Test with similar but not identical names
            segments = [{'text': 'Jon Smyth was present at the hearing'}]
            matches = checker.find_similar_terms(segments)

            assert len(matches) >= 1
            assert any(m.matched_term == 'John Smith' for m in matches)
            assert all(isinstance(m, FuzzyMatch) for m in matches)

    def test_phonetic_matching(self, mock_config, sample_sensitive_terms):
        """Test phonetic matching for names."""
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_open(read_data=sample_sensitive_terms)):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            # Test with phonetically similar names
            segments = [{'text': 'Robbert Joanes attended the meeting'}]
            matches = checker.find_similar_terms(segments)

            assert len(matches) >= 1
            assert any(m.matched_term == 'Robert Jones' for m in matches)
            assert any(m.match_type == 'phonetic' for m in matches)

    def test_address_matching(self, mock_config, sample_sensitive_terms):
        """Test matching of address patterns."""
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_open(read_data=sample_sensitive_terms)):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            # Test with similar address
            segments = [{'text': 'They lived at 123 Maine Street'}]
            matches = checker.find_similar_terms(segments)

            assert len(matches) >= 1
            assert any('123 Main Street' in m.matched_term for m in matches)

    def test_phone_number_matching(self, mock_config, sample_sensitive_terms):
        """Test matching of phone number patterns."""
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_open(read_data=sample_sensitive_terms)):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)

            # Test with similar phone number
            segments = [{'text': 'Contact number is +44 7700 900 123'}]
            matches = checker.find_similar_terms(segments)

            assert len(matches) >= 1
            assert any('+44 7700 900123' in m.matched_term for m in matches)

    def test_save_matches(self, mock_config, sample_sensitive_terms):
        """Test saving matches to JSON."""
        # First mock for loading sensitive terms
        mock_file = mock_open(read_data=sample_sensitive_terms)

        # Create a second mock for the output file
        output_mock = mock_open()

        # Combine the mocks
        mock_file.side_effect = [
            mock_open(read_data=sample_sensitive_terms)(),  # For loading terms
            output_mock()  # For saving matches
        ]

        with patch('pathlib.Path.exists') as mock_exists, \
                patch('builtins.open', mock_file):
            mock_exists.return_value = True
            checker = FuzzyChecker(mock_config)
            segments = [{'text': 'Jon Smyth and Robert Joanes were present'}]
            matches = checker.find_similar_terms(segments)

            # Test saving matches
            output_path = Path('test_output.json')
            checker.save_matches(matches, output_path)

            # Verify that write was called
            output_mock().write.assert_called()
            # Check that the written data is valid JSON and contains expected data
            write_call_args = output_mock().write.call_args[0][0]
            assert len(write_call_args) > 0
            # Verify we can parse it as JSON
            saved_data = json.loads(write_call_args)
            assert isinstance(saved_data, list)
            assert len(saved_data) > 0
            assert 'matched_term' in saved_data[0]

    def test_error_handling(self, mock_config):
        """Test error handling for missing files and invalid input."""
        # Test with non-existent sensitive terms file
        checker = FuzzyChecker(mock_config)
        assert len(checker.sensitive_terms) == 0  # Should handle missing file gracefully

        # Test with empty segments
        matches = checker.find_similar_terms([])
        assert len(matches) == 0

        # Test with None text
        matches = checker.find_similar_terms([{'text': None}])
        assert len(matches) == 0


if __name__ == '__main__':
    pytest.main([__file__])