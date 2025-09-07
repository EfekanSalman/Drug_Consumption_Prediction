"""
Training workflow for drug consumption prediction models with MLflow integration.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime

from preprocessing import DrugDataPreprocessor, prepare_target_variable
from utils import load_data, get_data_path, setup_logging, save_model


def setup_mlflow_experiment(experiment_name: str = "drug_consumption_prediction"):
    """
    Setup MLflow experiment for tracking.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        str: Experiment ID
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logging.info(f"Using existing MLflow experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logging.warning(f"MLflow setup failed: {e}. Continuing without MLflow tracking.")
        return None


def log_model_metrics(y_test, predictions, run_name: str = None):
    """
    Log model performance metrics to MLflow.

    Args:
        y_test: True target values
        predictions: Model predictions
        run_name (str, optional): Name for the MLflow run
    """
    try:
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        logging.info(f"Logged metrics - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    except Exception as e:
        logging.warning(f"Failed to log metrics: {e}")


def log_hyperparameters(best_params: dict):
    """
    Log hyperparameters to MLflow.

    Args:
        best_params (dict): Best hyperparameters from GridSearchCV
    """
    try:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logging.info(f"Logged hyperparameters: {best_params}")
    except Exception as e:
        logging.warning(f"Failed to log hyperparameters: {e}")


def log_model_artifact(model, model_name: str = "cannabis_model"):
    """
    Log model as MLflow artifact.

    Args:
        model: Trained model to log
        model_name (str): Name for the model artifact
    """
    try:
        mlflow.sklearn.log_model(model, model_name)
        logging.info(f"Logged model artifact: {model_name}")
    except Exception as e:
        logging.warning(f"Failed to log model artifact: {e}")


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
    Main training workflow with MLflow integration.

    Returns:
        tuple: (best_model, evaluation_results, feature_importance)
    """
    setup_logging()
    logging.info("Starting the drug consumption prediction training.")

    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment()

    # Start MLflow run
    with mlflow.start_run(run_name=f"cannabis_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # Log run metadata
            mlflow.log_param("target_drug", "Cannabis")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)

            # 1. Load the data
            file_path = get_data_path()
            df = load_data(file_path)

            if df is None:
                logging.error("Failed to load data. Exiting.")
                return None, None, None

            # Log dataset info
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("num_features", len(df.columns))

            # 2. Prepare the target variable
            target_df = prepare_target_variable(df.copy(), 'Cannabis', is_binary=True)
            if target_df is None:
                logging.error("Target variable preparation failed. Exiting.")
                return None, None, None

            features = df.drop(columns=target_df.columns, errors='ignore')
            target = target_df['Cannabis']

            # Log target distribution
            target_dist = target.value_counts()
            mlflow.log_param("target_class_0_count", int(target_dist.get(0, 0)))
            mlflow.log_param("target_class_1_count", int(target_dist.get(1, 0)))

            # 3. Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            logging.info("Data split into training and testing sets.")

            # 4. Train the model
            grid_search = train_model(X_train, y_train)
            best_model = grid_search.best_estimator_

            logging.info(f"Best parameters found: {grid_search.best_params_}")

            # Log hyperparameters
            log_hyperparameters(grid_search.best_params_)

            # 5. Evaluate the model
            evaluation_results = evaluate_model(best_model, X_test, y_test)

            # Log metrics to MLflow
            log_model_metrics(y_test, evaluation_results['predictions'])

            # 6. Analyze feature importance
            feature_importance = analyze_feature_importance(best_model, features)

            # Log top 5 feature importances
            top_features = feature_importance.head(5)
            for i, (feature, importance) in enumerate(top_features.items()):
                mlflow.log_metric(f"feature_importance_{i+1}_{feature}", importance)

            # 7. Save the best model
            model_path = get_data_path().parent / "cannabis_model.pkl"
            save_model(best_model, str(model_path))

            # Log model artifact
            log_model_artifact(best_model)

            # Log model path
            mlflow.log_param("model_save_path", str(model_path))

            logging.info("Training completed successfully with MLflow tracking.")

            return best_model, evaluation_results, feature_importance

        except Exception as e:
            logging.error(f"Training failed: {e}")
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    main()
