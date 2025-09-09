"""
Utility functions for the drug consumption prediction project.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    return Path(__file__).parent.parent


def get_data_path(filename: str = "Drug_Consumption.csv") -> Path:
    """
    Get the path to data file.
    
    Args:
        filename: Name of the data file
        
    Returns:
        Path object pointing to data file
    """
    return get_project_root() / "data" / filename


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def save_model(model, filepath: str) -> bool:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import joblib
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return False


def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model if successful, None otherwise
    """
    try:
        import joblib
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
