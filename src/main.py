import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
from pathlib import Path

from preprocessing import DrugDataPreprocessor, prepare_target_variable


def main():
    """
    Main function to orchestrate the data preprocessing, model training,
    and evaluation workflow for predicting Cannabis use.
    """
    logging.info("Starting the drug consumption prediction project.")

    # 1. Initialize the preprocessor and load the data
    # Create a robust path that works regardless of the current working directory.
    # It navigates up one level from 'src' to the project root, then down to 'data'.
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "Drug_Consumption.csv"

    preprocessor = DrugDataPreprocessor(file_path)
    preprocessor.load_data()

    if preprocessor.df is None:
        logging.error("Data could not be loaded. Exiting.")
        return

    # 2. Preprocess features and prepare the target variable
    # Fit and transform the features using the scikit-learn compatible preprocessor
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

    # 4. Train a classification model (Random Forest is a good starting point)
    logging.info("Training a RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # 5. Evaluate the model's performance on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Model Accuracy on the test set: {accuracy:.2f}")

    # 6. Analyze Feature Importance (Bonus Step)
    # This helps in understanding which features contributed most to the prediction
    feature_importances = pd.Series(model.feature_importances_, index=features.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 5 Most Important Features for Cannabis Prediction:")
    print(sorted_importances.head())


if __name__ == "__main__":
    main()
