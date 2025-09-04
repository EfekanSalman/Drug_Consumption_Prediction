import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt


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


def main():
    """
    Main function to orchestrate the data preprocessing, model training,
    and evaluation workflow with hyperparameter tuning and model saving.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the drug consumption prediction project.")

    # 1. Load the data
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "Drug_Consumption.csv"

    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}. Exiting.")
        return

    # 2. Prepare the target variable
    target_df = prepare_target_variable(df.copy(), 'Cannabis', is_binary=True)
    if target_df is None:
        logging.error("Target variable preparation failed. Exiting.")
        return

    features = df.drop(columns=target_df.columns, errors='ignore')
    target = target_df['Cannabis']

    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    logging.info("Data split into training and testing sets.")

    # 4. Create a pipeline with the preprocessor, SMOTE, and the classifier
    logging.info("Creating a pipeline for hyperparameter tuning...")
    pipeline = Pipeline([
        ('preprocessor', DrugDataPreprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 5. Define the parameter grid for the GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    }

    # 6. Perform a grid search to find the best parameters
    logging.info("Performing GridSearchCV to find the best model...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logging.info("Grid search completed.")

    # 7. Get the best model and evaluate its performance
    best_model = grid_search.best_estimator_

    logging.info(f"Best parameters found: {grid_search.best_params_}")

    predictions = best_model.predict(X_test)
    accuracy = best_model.score(X_test, y_test)

    logging.info(f"Model Accuracy on the test set: {accuracy:.2f}")

    # Display the full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 8. Visualize the Confusion Matrix
    print("\nConfusion Matrix:")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap=plt.cm.Blues)
    plt.show()

    # 9. Save the best model for later use
    model_path = project_root / "cannabis_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Best model saved to {model_path}")

    # 10. Analyze Feature Importance
    # Get feature importances from the classifier step of the best pipeline
    model = best_model.named_steps['classifier']
    preprocessor = best_model.named_steps['preprocessor']

    # Note: Feature importance is from the classifier's perspective,
    # so we need to get the feature names after preprocessing.
    preprocessed_features = preprocessor.transform(features)
    feature_importances = pd.Series(model.feature_importances_, index=preprocessed_features.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 5 Most Important Features for Cannabis Prediction:")
    print(sorted_importances.head())


if __name__ == "__main__":
    main()
