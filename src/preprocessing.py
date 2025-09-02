import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import numpy as np
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DrugDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class to handle data loading and preprocessing
    for the Drug Consumption dataset.
    This includes handling categorical variables and preparing features for modeling.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        # We will now encode both 'Ethnicity' and 'Country'
        self.categorical_features = ['Ethnicity', 'Country']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.drug_columns = [
            'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
            'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
            'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
        ]

    def load_data(self) -> None:
        """
        Loads the raw CSV data into a pandas DataFrame.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully.")
            if 'ID' in self.df.columns:
                self.df.set_index('ID', inplace=True)
        except FileNotFoundError:
            logging.error(f"Error: File not found at {self.file_path}")
            self.df = None

    def fit(self, X: pd.DataFrame, y=None) -> 'DrugDataPreprocessor':
        """
        Fits the preprocessor on the data to learn categorical encodings.
        """
        if X is None or X.empty:
            logging.warning("Input DataFrame is empty or None. Cannot fit.")
            return self

        # Fit the OneHotEncoder on the specified categorical columns
        categorical_data = X[self.categorical_features]
        self.encoder.fit(categorical_data)
        return self

    def transform(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transforms the data by handling categorical variables and dropping drug columns.
        """
        if X is None or X.empty:
            logging.warning("Input DataFrame is empty or None. Cannot transform.")
            return None

        # Make a copy to avoid modifying the original DataFrame
        df_copy = X.copy()

        # Handle 'Age' and 'Education' with label encoding
        age_map = {
            '18-24': 0, '25-34': 1, '35-44': 2,
            '45-54': 3, '55-64': 4, '65+': 5
        }
        df_copy['Age'] = df_copy['Age'].map(age_map).fillna(-1)

        edu_map = {
            'Left school at 16': 0, 'Left school at 17': 1, 'Left school at 18': 2,
            'Some college or university, no degree': 3, 'Professional Certificate/ Diploma': 4,
            'University Degree': 5, 'Masters Degree': 6, 'Doctorate Degree': 7
        }
        df_copy['Education'] = df_copy['Education'].map(edu_map).fillna(-1)

        # Handle 'Gender' with mapping and NaN fill
        gender_map = {'Male': 0, 'Female': 1}
        df_copy['Gender'] = df_copy['Gender'].map(gender_map).fillna(-1)

        # Apply One-Hot Encoding for specified categorical features
        try:
            encoded_features = self.encoder.transform(df_copy[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_features,
                                      columns=self.encoder.get_feature_names_out(self.categorical_features),
                                      index=df_copy.index)
            df_copy = pd.concat([df_copy.drop(columns=self.categorical_features), encoded_df], axis=1)
        except Exception as e:
            logging.error(f"Error during OneHotEncoding: {e}")
            return None

        # Drop drug columns to prepare feature set
        try:
            features_df = df_copy.drop(columns=self.drug_columns)
        except KeyError as e:
            logging.error(f"Failed to drop drug columns. Check if all drug columns are present: {e}")
            return None

        logging.info("Features preprocessed successfully.")
        return features_df

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Returns the preprocessed DataFrame.
        """
        return self.df.copy() if self.df is not None else None


def prepare_target_variable(df: pd.DataFrame, drug_name: str, is_binary: bool = True) -> Optional[pd.DataFrame]:
    """
    Prepares a specific drug column as a target variable, either binary or multi-class.

    Args:
        df (pd.DataFrame): The DataFrame containing the drug consumption data.
        drug_name (str): The name of the drug column (e.g., 'Alcohol', 'Coke').
        is_binary (bool): If True, converts the target to binary (user/non-user).
                         If False, keeps the original multi-class labels.

    Returns:
        pd.DataFrame: A DataFrame with the specified target column converted.
    """
    df_copy = df.copy()
    if drug_name not in df_copy.columns:
        logging.error(f"Drug column '{drug_name}' not found in the DataFrame.")
        return None

    if is_binary:
        # Assuming 'CL0' is the non-user group, and all others are users.
        df_copy[drug_name] = df_copy[drug_name].apply(lambda x: 0 if x == 'CL0' else 1)
    # If not binary, we'll assume a multi-class mapping is desired.
    else:
        multi_class_map = {'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6}
        df_copy[drug_name] = df_copy[drug_name].map(multi_class_map).fillna(-1)

    logging.info(f"Target variable for '{drug_name}' prepared successfully.")
    return df_copy
