"""
Unit tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from preprocessing import DrugDataPreprocessor, prepare_target_variable


class TestDrugDataPreprocessor:
    """Test cases for DrugDataPreprocessor class."""

    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = DrugDataPreprocessor()

        assert preprocessor.drug_columns is not None
        assert preprocessor.categorical_features is not None
        assert preprocessor.learned_features is None
        assert 'Cannabis' in preprocessor.drug_columns
        assert 'Country' in preprocessor.categorical_features
        assert 'Ethnicity' in preprocessor.categorical_features

    def test_fit(self, sample_drug_data, preprocessor):
        """Test the fit method."""
        # Remove target columns for fitting
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])

        result = preprocessor.fit(features)

        # Should return self for method chaining
        assert result is preprocessor
        # Should have learned features
        assert preprocessor.learned_features is not None
        assert len(preprocessor.learned_features) > 0

    def test_fit_transform_consistency(self, sample_drug_data, preprocessor):
        """Test that fit and transform produce consistent results."""
        # Remove target columns for fitting
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])

        # Fit on training data
        preprocessor.fit(features)
        train_transformed = preprocessor.transform(features)

        # Transform same data should produce same result
        train_transformed_2 = preprocessor.transform(features)

        pd.testing.assert_frame_equal(train_transformed, train_transformed_2)

    def test_transform_without_fit(self, sample_drug_data, preprocessor):
        """Test that transform raises error if fit hasn't been called."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])

        with pytest.raises(ValueError, match="Must call fit\\(\\) before transform\\(\\)"):
            preprocessor.transform(features)

    def test_gender_mapping(self, sample_drug_data, fitted_preprocessor):
        """Test gender mapping to numerical values."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # Check that Gender column exists and has numerical values
        assert 'Gender' in transformed.columns
        assert transformed['Gender'].dtype in [np.int64, np.float64]
        # Should not have any NaN values after transformation
        assert not transformed['Gender'].isna().any()

    def test_education_mapping(self, sample_drug_data, fitted_preprocessor):
        """Test education mapping to numerical values."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # Check that Education column exists and has numerical values
        assert 'Education' in transformed.columns
        assert transformed['Education'].dtype in [np.int64, np.float64]
        # Should not have any NaN values after transformation
        assert not transformed['Education'].isna().any()

    def test_categorical_encoding(self, sample_drug_data, fitted_preprocessor):
        """Test one-hot encoding of categorical features."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # Check that categorical features are one-hot encoded
        # Should have columns like 'Country_UK', 'Country_USA', etc.
        country_cols = [col for col in transformed.columns if col.startswith('Country_')]
        ethnicity_cols = [col for col in transformed.columns if col.startswith('Ethnicity_')]

        assert len(country_cols) > 0, "Country should be one-hot encoded"
        assert len(ethnicity_cols) > 0, "Ethnicity should be one-hot encoded"

    def test_drug_columns_removed(self, sample_drug_data, fitted_preprocessor):
        """Test that drug columns are removed from features."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # Drug columns should not be in transformed data
        for drug_col in fitted_preprocessor.drug_columns:
            if drug_col in features.columns:
                assert drug_col not in transformed.columns, f"{drug_col} should be removed"

    def test_irrelevant_columns_removed(self, sample_drug_data, fitted_preprocessor):
        """Test that irrelevant columns (ID, Age) are removed."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # ID and Age should not be in transformed data
        assert 'ID' not in transformed.columns
        assert 'Age' not in transformed.columns

    def test_all_numeric_output(self, sample_drug_data, fitted_preprocessor):
        """Test that all output columns are numeric."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # All columns should be numeric
        for col in transformed.columns:
            assert transformed[col].dtype in [np.int64, np.float64], f"Column {col} should be numeric"

    def test_no_nan_values(self, sample_drug_data, fitted_preprocessor):
        """Test that output has no NaN values."""
        features = sample_drug_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])
        transformed = fitted_preprocessor.transform(features)

        # Should not have any NaN values
        assert not transformed.isna().any().any()

    def test_new_data_handling(self, sample_drug_data, fitted_preprocessor):
        """Test handling of new data with different categorical values."""
        # Create new data with different categorical values
        new_data = sample_drug_data.copy()
        new_data['Country'] = 'France'  # New country not in training data
        new_data['Ethnicity'] = 'Other'  # New ethnicity not in training data
        features = new_data.drop(columns=['Cannabis', 'Coke', 'Amphet'])

        # Should not raise error and should handle new categories gracefully
        transformed = fitted_preprocessor.transform(features)

        # Should have same number of columns as training data
        assert len(transformed.columns) == len(fitted_preprocessor.learned_features)
        # Should not have any NaN values
        assert not transformed.isna().any().any()


class TestPrepareTargetVariable:
    """Test cases for prepare_target_variable function."""

    def test_prepare_target_variable_binary(self, sample_drug_data):
        """Test binary target variable preparation."""
        target_df = prepare_target_variable(sample_drug_data, 'Cannabis', is_binary=True)

        assert target_df is not None
        assert 'Cannabis' in target_df.columns
        assert len(target_df) == len(sample_drug_data)

        # Check binary mapping
        unique_values = target_df['Cannabis'].unique()
        assert set(unique_values).issubset({0, 1}), "Should only contain 0 and 1"

    def test_prepare_target_variable_non_binary(self, sample_drug_data):
        """Test non-binary target variable preparation."""
        target_df = prepare_target_variable(sample_drug_data, 'Cannabis', is_binary=False)

        assert target_df is not None
        assert 'Cannabis' in target_df.columns
        assert len(target_df) == len(sample_drug_data)

        # Should contain original values
        original_values = sample_drug_data['Cannabis'].unique()
        target_values = target_df['Cannabis'].unique()
        assert set(target_values) == set(original_values)

    def test_prepare_target_variable_invalid_drug(self, sample_drug_data):
        """Test handling of invalid drug name."""
        target_df = prepare_target_variable(sample_drug_data, 'InvalidDrug', is_binary=True)

        assert target_df is None

    def test_binary_mapping_correctness(self, sample_drug_data):
        """Test that binary mapping is correct."""
        target_df = prepare_target_variable(sample_drug_data, 'Cannabis', is_binary=True)

        # Check specific mappings
        for idx, row in sample_drug_data.iterrows():
            original_value = row['Cannabis']
            binary_value = target_df.loc[idx, 'Cannabis']

            if original_value in ['CL0', 'CL1']:
                assert binary_value == 0, f"CL0/CL1 should map to 0, got {binary_value}"
            else:
                assert binary_value == 1, f"CL2+ should map to 1, got {binary_value}"
