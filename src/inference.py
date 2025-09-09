"""
Inference module for making predictions on new data using trained models.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

from utils import load_model, get_data_path, setup_logging


class DrugConsumptionPredictor:
    """
    A class to handle inference using trained drug consumption prediction models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file. If None, uses default path.
        """
        if model_path is None:
            model_path = str(get_data_path().parent / "cannabis_model.pkl")
        
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        self.model = load_model(self.model_path)
        if self.model is None:
            logging.error(f"Failed to load model from {self.model_path}")
            return False
        return True
    
    def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[pd.Series]:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame or dictionary with input features
            
        Returns:
            Predictions as pandas Series, or None if prediction failed
        """
        if self.model is None:
            logging.error("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Make predictions
            predictions = self.model.predict(data)
            
            # Convert to Series for consistency
            if isinstance(predictions, (list, tuple)):
                predictions = pd.Series(predictions)
            
            logging.info(f"Successfully made {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return None
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        Get prediction probabilities for new data.
        
        Args:
            data: DataFrame or dictionary with input features
            
        Returns:
            Prediction probabilities as DataFrame, or None if prediction failed
        """
        if self.model is None:
            logging.error("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(data)
            
            # Convert to DataFrame with class labels
            proba_df = pd.DataFrame(
                probabilities, 
                columns=[f'class_{i}' for i in range(probabilities.shape[1])]
            )
            
            logging.info(f"Successfully computed probabilities for {len(proba_df)} samples")
            return proba_df
            
        except Exception as e:
            logging.error(f"Error computing prediction probabilities: {e}")
            return None
    
    def get_feature_names(self) -> Optional[list]:
        """
        Get the feature names expected by the model.
        
        Returns:
            List of feature names, or None if model not loaded
        """
        if self.model is None:
            return None
        
        try:
            # Get the preprocessor to access learned features
            preprocessor = self.model.named_steps['preprocessor']
            return preprocessor.learned_features
        except Exception as e:
            logging.error(f"Error getting feature names: {e}")
            return None


def predict_single_sample(sample_data: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to predict on a single sample.
    
    Args:
        sample_data: Dictionary with feature values for a single sample
        model_path: Path to the saved model file
        
    Returns:
        Dictionary with prediction results
    """
    predictor = DrugConsumptionPredictor(model_path)
    
    if predictor.model is None:
        return {"error": "Failed to load model"}
    
    prediction = predictor.predict(sample_data)
    probabilities = predictor.predict_proba(sample_data)
    
    if prediction is None or probabilities is None:
        return {"error": "Failed to make prediction"}
    
    return {
        "prediction": int(prediction.iloc[0]) if len(prediction) > 0 else None,
        "probabilities": probabilities.iloc[0].to_dict() if len(probabilities) > 0 else None,
        "prediction_label": "High Risk" if prediction.iloc[0] == 1 else "Low Risk"
    }


def predict_batch(data: pd.DataFrame, model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to predict on a batch of samples.
    
    Args:
        data: DataFrame with feature values for multiple samples
        model_path: Path to the saved model file
        
    Returns:
        DataFrame with prediction results
    """
    predictor = DrugConsumptionPredictor(model_path)
    
    if predictor.model is None:
        return pd.DataFrame({"error": ["Failed to load model"] * len(data)})
    
    predictions = predictor.predict(data)
    probabilities = predictor.predict_proba(data)
    
    if predictions is None or probabilities is None:
        return pd.DataFrame({"error": ["Failed to make prediction"] * len(data)})
    
    # Combine results
    results = data.copy()
    results['prediction'] = predictions
    results['prediction_label'] = predictions.apply(lambda x: "High Risk" if x == 1 else "Low Risk")
    
    # Add probability columns
    for col in probabilities.columns:
        results[f'prob_{col}'] = probabilities[col]
    
    return results


def main():
    """
    Example usage of the inference module.
    """
    setup_logging()
    
    # Example: Load some test data and make predictions
    data_path = get_data_path()
    
    try:
        # Load a small sample of data for testing
        df = pd.read_csv(data_path)
        test_sample = df.iloc[0:5].drop(columns=['Cannabis'])  # Remove target column
        
        logging.info("Making predictions on test sample...")
        results = predict_batch(test_sample)
        
        print("\nPrediction Results:")
        print(results[['prediction', 'prediction_label']].head())
        
    except Exception as e:
        logging.error(f"Error in main inference example: {e}")


if __name__ == "__main__":
    main()
