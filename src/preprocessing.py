import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging


class DrugDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to integrate data preprocessing into the sklearn Pipeline.

    This transformer handles the preprocessing of drug consumption data including:
    - Gender and Education mapping to numerical values
    - One-hot encoding of categorical features (Country, Ethnicity)
    - Removal of irrelevant columns (ID, Age, drug columns)
    - Ensuring all features are numeric

    Attributes:
        drug_columns (list): List of drug-related column names to be excluded from features
        categorical_features (list): List of categorical features to be one-hot encoded
        learned_features (list): Feature names learned during fit() for consistent transform()
    """

    def __init__(self):
        """
        Initialize the DrugDataPreprocessor.

        Sets up the column mappings and initializes learned_features to None.
        """
        self.drug_columns = [
            'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
            'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine',
            'Semer', 'VSA'
        ]
        self.categorical_features = ['Country', 'Ethnicity']
        self.learned_features = None

    def fit(self, X, y=None):
        """
        Learn the feature names (columns) after one-hot encoding on the training data.

        Args:
            X (pd.DataFrame): Training data with raw features
            y (array-like, optional): Target values (not used in this transformer)

        Returns:
            self: Returns self for method chaining
        """
        temp_df = self._preprocess_step(X)
        self.learned_features = temp_df.columns.tolist()
        return self

    def transform(self, X):
        """
        Apply the preprocessing steps and ensure all columns match the learned features.

        Args:
            X (pd.DataFrame): Data to transform

        Returns:
            pd.DataFrame: Preprocessed data with consistent feature columns

        Raises:
            ValueError: If fit() has not been called before transform()
        """
        if self.learned_features is None:
            raise ValueError("Must call fit() before transform()")

        df = self._preprocess_step(X.copy())

        # Reindex to ensure all columns from the training set are present,
        # filling missing columns with 0.
        df = df.reindex(columns=self.learned_features, fill_value=0)

        return df

    def _preprocess_step(self, df):
        """
        Helper method to perform the core preprocessing steps.

        Args:
            df (pd.DataFrame): Input dataframe to preprocess

        Returns:
            pd.DataFrame: Preprocessed dataframe with:
                - Gender mapped to numerical values (F=0, M=1, unknown=-1)
                - Education mapped to numerical values (1-9 scale, unknown=-1)
                - Categorical features one-hot encoded
                - Irrelevant columns removed
                - All features converted to numeric
        """
        # Mapping 'Gender' to numerical values and filling missing values
        gender_map = {'F': 0, 'M': 1}
        df['Gender'] = df['Gender'].map(gender_map).fillna(-1)

        # Mapping 'Education' to numerical values and handling unexpected values
        education_map = {
            'Left school before 16 years': 1, 'Left school at 16 years': 2,
            'Left school at 17 years': 3, 'Left school at 18 years': 4,
            'Some college or university, no certificate or degree': 5,
            'Professional certificate/diploma': 6,
            'University degree': 7, 'Masters degree': 8, 'Doctorate degree': 9
        }
        df['Education'] = df['Education'].map(education_map).fillna(-1)

        # One-hot encoding for categorical features
        df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)

        # Drop irrelevant columns
        irrelevant_columns = ['ID', 'Age'] + self.drug_columns
        df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns], errors='ignore')

        # Ensure all feature columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Final check to fill any remaining NaN values before returning
        df = df.fillna(0)

        return df


def prepare_target_variable(df, target_drug, is_binary=True):
    """
    Prepares the target variable for a specific drug from the dataset.

    This function extracts and optionally transforms the target drug column for machine learning.
    For binary classification, it maps consumption levels to binary values:
    - CL0, CL1 (never used, used over a decade ago) → 0 (Low Risk)
    - CL2, CL3, CL4, CL5, CL6 (used in last year, month, week, day, last day) → 1 (High Risk)

    Args:
        df (pd.DataFrame): Input dataframe containing drug consumption data
        target_drug (str): Name of the drug column to use as target variable
        is_binary (bool, optional): Whether to convert to binary classification. Defaults to True.

    Returns:
        pd.DataFrame or None: DataFrame with target variable, or None if target_drug not found

    Example:
        >>> df = pd.read_csv('drug_data.csv')
        >>> target = prepare_target_variable(df, 'Cannabis', is_binary=True)
        >>> print(target['Cannabis'].value_counts())
    """
    if target_drug not in df.columns:
        logging.error(f"Target drug '{target_drug}' not found in data.")
        return None

    target_df = df[[target_drug]].copy()

    if is_binary:
        # Binary classification for 'CL0', 'CL1' vs 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'
        target_df[target_drug] = target_df[target_drug].apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1)

    return target_df


