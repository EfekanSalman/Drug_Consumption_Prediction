import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging


# Custom transformer to integrate data preprocessing into the sklearn Pipeline.
class DrugDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
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
        """
        temp_df = self._preprocess_step(X)
        self.learned_features = temp_df.columns.tolist()
        return self

    def transform(self, X):
        """
        Apply the preprocessing steps and ensure all columns match the learned features.
        """
        df = self._preprocess_step(X.copy())

        # Reindex to ensure all columns from the training set are present,
        # filling missing columns with 0.
        df = df.reindex(columns=self.learned_features, fill_value=0)

        return df

    def _preprocess_step(self, df):
        """
        Helper method to perform the core preprocessing steps.
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
    Prepares the target variable for a specific drug.
    """
    if target_drug not in df.columns:
        logging.error(f"Target drug '{target_drug}' not found in data.")
        return None

    target_df = df[[target_drug]].copy()

    if is_binary:
        # Binary classification for 'CL0', 'CL1' vs 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'
        target_df[target_drug] = target_df[target_drug].apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1)

    return target_df


