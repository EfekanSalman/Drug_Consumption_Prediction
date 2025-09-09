"""
Pytest configuration and shared fixtures for the drug consumption prediction tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import DrugDataPreprocessor, prepare_target_variable


@pytest.fixture
def sample_drug_data():
    """
    Create sample drug consumption data for testing.
    
    Returns:
        pd.DataFrame: Sample dataframe with drug consumption data
    """
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Age': [25, 30, 35, 28, 32],
        'Gender': ['F', 'M', 'F', 'M', 'F'],
        'Education': ['University degree', 'Masters degree', 'Left school at 18 years', 
                     'Professional certificate/diploma', 'Doctorate degree'],
        'Country': ['USA', 'UK', 'Canada', 'USA', 'Australia'],
        'Ethnicity': ['White', 'Asian', 'Black', 'White', 'Hispanic'],
        'Alcohol': ['CL0', 'CL1', 'CL2', 'CL3', 'CL4'],
        'Cannabis': ['CL0', 'CL1', 'CL2', 'CL3', 'CL4'],
        'Coke': ['CL0', 'CL0', 'CL1', 'CL2', 'CL3'],
        'Amphet': ['CL0', 'CL0', 'CL0', 'CL1', 'CL2']
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """
    Create a DrugDataPreprocessor instance for testing.
    
    Returns:
        DrugDataPreprocessor: Instance of the preprocessor
    """
    return DrugDataPreprocessor()


@pytest.fixture
def fitted_preprocessor(sample_drug_data, preprocessor):
    """
    Create a fitted DrugDataPreprocessor for testing.
    
    Args:
        sample_drug_data: Sample data fixture
        preprocessor: Preprocessor fixture
        
    Returns:
        DrugDataPreprocessor: Fitted preprocessor
    """
    # Remove target columns for fitting
    features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
    preprocessor.fit(features)
    return preprocessor

