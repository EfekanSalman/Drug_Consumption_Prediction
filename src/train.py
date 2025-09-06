"""
Training workflow for drug consumption prediction models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from preprocessing import DrugDataPreprocessor, prepare_target_variable
from utils import load_data, get_data_path, setup_logging, save_model


def create_pipeline():
    """
    Create a pipeline with preprocessor, SMOTE, and classifier.

    Returns:
        Pipeline object
    """
    return Pipeline([
        ('preprocessor', DrugDataPreprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])


def get_hyperparameter_grid():
    """
    Define hyperparameter grid for GridSearchCV.

    Returns:
        Dictionary with parameter grid
    """
    return {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    }


def train_model(X_train, y_train, param_grid=None):
    """
    Train a model using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid for hyperparameter tuning

    Returns:
        Best model from GridSearchCV
    """
    if param_grid is None:
        param_grid = get_hyperparameter_grid()

    pipeline = create_pipeline()

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

    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Model Accuracy on the test set: {accuracy:.2f}")

    # Display the full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

    # Visualize the Confusion Matrix
    print("\nConfusion Matrix:")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap=plt.cm.Blues)
    # For the Linux
    #plt.savefig("confusion_matrix.png")
    plt.show()

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'classification_report': classification_report(y_test, predictions)
    }


def analyze_feature_importance(model, features):
    """
    Analyze and display feature importance.

    Args:
        model: Trained model with feature importance
        features: Original features DataFrame
    """
    # Get feature importances from the classifier step of the best pipeline
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # Get feature names after preprocessing
    preprocessed_features = preprocessor.transform(features)
    feature_importances = pd.Series(
        classifier.feature_importances_,
        index=preprocessed_features.columns
    )
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 5 Most Important Features for Cannabis Prediction:")
    print(sorted_importances.head())

    return sorted_importances


def main():
    """
    Main training workflow.
    """
    setup_logging()
    logging.info("Starting the drug consumption prediction training.")

    # 1. Load the data
    file_path = get_data_path()
    df = load_data(file_path)

    if df is None:
        logging.error("Failed to load data. Exiting.")
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

    # 4. Train the model
    grid_search = train_model(X_train, y_train)
    best_model = grid_search.best_estimator_

    logging.info(f"Best parameters found: {grid_search.best_params_}")

    # 5. Evaluate the model
    evaluation_results = evaluate_model(best_model, X_test, y_test)

    # 6. Analyze feature importance
    feature_importance = analyze_feature_importance(best_model, features)

    # 7. Save the best model
    model_path = get_data_path().parent / "cannabis_model.pkl"
    save_model(best_model, str(model_path))

    return best_model, evaluation_results, feature_importance


if __name__ == "__main__":
    main()
