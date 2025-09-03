import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import logging
from pathlib import Path

from preprocessing import DrugDataPreprocessor, prepare_target_variable

def main():
    """
    Main function to orchestrate the data preprocessing, model training,
    and evaluation workflow for predicting Cannabis use with hyperparameter tuning.
    """
    logging.info("Starting the drug consumption prediction project.")

    # 1. Initialize the preprocessor and load the data
    # Create a robust path that works regardless of the current working directory.
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "Drug_Consumption.csv"

    preprocessor = DrugDataPreprocessor(file_path)
    preprocessor.load_data()

    if preprocessor.df is None:
        logging.error("Data could not be loaded. Exiting.")
        return

    # 2. Preprocess features and prepare the target variable
    preprocessor.fit(preprocessor.df)
    features_df = preprocessor.transform(preprocessor.df)
    # Use a copy of the original DataFrame to prepare the target variable.
    target_df = prepare_target_variable(preprocessor.df.copy(), 'Cannabis', is_binary=True)

    if features_df is None or target_df is None:
        logging.error("Data preprocessing failed. Exiting.")
        return

    # Align features and target to ensure they have the same indices
    features = features_df
    target = target_df['Cannabis']

    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    logging.info("Data split into training and testing sets.")

    # 4. Create a pipeline with SMOTE and the classifier
    logging.info("Creating a pipeline for hyperparameter tuning...")
    pipeline = Pipeline([
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
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    logging.info("Grid search completed.")

    # 7. Get the best model and evaluate its performance
    best_model = grid_search.best_estimator_

    logging.info(f"Best parameters found: {grid_search.best_params_}")

    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model Accuracy on the test set: {accuracy:.2f}")

    # Display the full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 8. Analyze Feature Importance
    model = best_model.named_steps['classifier']
    feature_importances = pd.Series(model.feature_importances_, index=features.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 5 Most Important Features for Cannabis Prediction:")
    print(sorted_importances.head())


if __name__ == "__main__":
    main()
