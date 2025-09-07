"""
Unit tests for the utils module.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from utils import (
    load_data, get_project_root, get_data_path,
    setup_logging, save_model, load_model
)


class TestDataLoading:
    """Test cases for data loading utilities."""

    def test_load_data_success(self):
        """Test successful data loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name

        try:
            df = load_data(temp_path)
            assert df is not None
            assert len(df) == 2
            assert list(df.columns) == ['col1', 'col2']
        finally:
            os.unlink(temp_path)

    def test_load_data_file_not_found(self):
        """Test handling of file not found error."""
        df = load_data("nonexistent_file.csv")
        assert df is None

    def test_load_data_invalid_file(self):
        """Test handling of invalid file format."""
        # Create temporary file with invalid content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith,missing,quotes\n")
            temp_path = f.name

        try:
            # This should still work as pandas is quite forgiving
            df = load_data(temp_path)
            assert df is not None
        finally:
            os.unlink(temp_path)


class TestPathUtilities:
    """Test cases for path utility functions."""

    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        # Should contain src directory
        assert (root / "src").exists()

    def test_get_data_path_default(self):
        """Test getting default data path."""
        data_path = get_data_path()
        assert isinstance(data_path, Path)
        assert data_path.name == "Drug_Consumption.csv"
        assert data_path.parent.name == "data"

    def test_get_data_path_custom(self):
        """Test getting custom data path."""
        custom_filename = "custom_data.csv"
        data_path = get_data_path(custom_filename)
        assert isinstance(data_path, Path)
        assert data_path.name == custom_filename
        assert data_path.parent.name == "data"


class TestLogging:
    """Test cases for logging utilities."""

    def test_setup_logging(self):
        """Test logging setup."""
        # This should not raise an error
        setup_logging()
        # We can't easily test the logging configuration without more complex setup
        assert True


class TestModelUtilities:
    """Test cases for model saving/loading utilities."""

    def test_save_model_success(self):
        """Test successful model saving."""

        # Create a simple mock model
        class MockModel:
            def __init__(self):
                self.fitted = True

        model = MockModel()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Mock joblib.dump to avoid actual file operations
            with patch('utils.joblib.dump') as mock_dump:
                result = save_model(model, temp_path)
                assert result is True
                mock_dump.assert_called_once_with(model, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_model_failure(self):
        """Test model saving failure."""
        model = "invalid_model"

        with patch('utils.joblib.dump', side_effect=Exception("Save failed")):
            result = save_model(model, "test_path.pkl")
            assert result is False

    def test_load_model_success(self):
        """Test successful model loading."""

        # Create a simple mock model
        class MockModel:
            def __init__(self):
                self.fitted = True

        expected_model = MockModel()

        with patch('utils.joblib.load', return_value=expected_model):
            model = load_model("test_path.pkl")
            assert model is expected_model

    def test_load_model_failure(self):
        """Test model loading failure."""
        with patch('utils.joblib.load', side_effect=Exception("Load failed")):
            model = load_model("test_path.pkl")
            assert model is None
