"""
Integration tests for the FastAPI application.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app
from api.models import PredictionRequest, GenderEnum, EducationEnum, CountryEnum, EthnicityEnum


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_request():
    """Create a sample prediction request."""
    return {
        "age": 25,
        "gender": "F",
        "education": "University degree",
        "country": "USA",
        "ethnicity": "White",
        "nscore": 0.5,
        "escore": -0.2,
        "oscore": 1.2,
        "ascore": 0.8,
        "cscore": -0.1,
        "impulsiveness": 0.3,
        "sensation_seeking": 1.0
    }


@pytest.fixture
def mock_predictor():
    """Create a mock predictor."""
    mock = MagicMock()
    mock.model = MagicMock()  # Simulate loaded model
    mock.predict.return_value = pd.Series([0])  # Mock prediction
    mock.predict_proba.return_value = pd.DataFrame({
        'class_0': [0.7],
        'class_1': [0.3]
    })
    mock.get_feature_names.return_value = ['feature1', 'feature2', 'feature3']
    return mock


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
    
    @patch('api.main.predictor')
    def test_predict_endpoint_success(self, mock_predictor_attr, client, sample_prediction_request, mock_predictor):
        """Test successful prediction endpoint."""
        mock_predictor_attr.return_value = mock_predictor
        
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "probabilities" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "prediction_id" in data
        
        # Verify prediction values
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["Low Risk", "High Risk"]
        assert isinstance(data["probabilities"], dict)
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_endpoint_validation_error(self, client):
        """Test prediction endpoint with validation error."""
        invalid_request = {
            "age": 15,  # Invalid age
            "gender": "F",
            "education": "University degree",
            "country": "USA",
            "ethnicity": "White",
            "nscore": 0.5,
            "escore": -0.2,
            "oscore": 1.2,
            "ascore": 0.8,
            "cscore": -0.1,
            "impulsiveness": 0.3,
            "sensation_seeking": 1.0
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_request = {
            "age": 25,
            "gender": "F"
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.predictor')
    def test_batch_predict_endpoint_success(self, mock_predictor_attr, client, mock_predictor):
        """Test successful batch prediction endpoint."""
        mock_predictor_attr.return_value = mock_predictor
        mock_predictor.predict.return_value = pd.Series([0, 1])  # Two predictions
        mock_predictor.predict_proba.return_value = pd.DataFrame({
            'class_0': [0.7, 0.3],
            'class_1': [0.3, 0.7]
        })
        
        batch_request = {
            "predictions": [
                {
                    "age": 25,
                    "gender": "F",
                    "education": "University degree",
                    "country": "USA",
                    "ethnicity": "White",
                    "nscore": 0.5,
                    "escore": -0.2,
                    "oscore": 1.2,
                    "ascore": 0.8,
                    "cscore": -0.1,
                    "impulsiveness": 0.3,
                    "sensation_seeking": 1.0
                },
                {
                    "age": 30,
                    "gender": "M",
                    "education": "Masters degree",
                    "country": "UK",
                    "ethnicity": "Asian",
                    "nscore": -0.5,
                    "escore": 0.2,
                    "oscore": -1.2,
                    "ascore": -0.8,
                    "cscore": 0.1,
                    "impulsiveness": -0.3,
                    "sensation_seeking": -1.0
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_predictions" in data
        assert "processing_time_ms" in data
        assert len(data["predictions"]) == 2
        assert data["total_predictions"] == 2
    
    def test_batch_predict_endpoint_empty_list(self, client):
        """Test batch prediction endpoint with empty list."""
        batch_request = {"predictions": []}
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint_too_many_items(self, client):
        """Test batch prediction endpoint with too many items."""
        # Create a list with 101 items (exceeds max of 100)
        predictions = []
        for i in range(101):
            predictions.append({
                "age": 25,
                "gender": "F",
                "education": "University degree",
                "country": "USA",
                "ethnicity": "White",
                "nscore": 0.5,
                "escore": -0.2,
                "oscore": 1.2,
                "ascore": 0.8,
                "cscore": -0.1,
                "impulsiveness": 0.3,
                "sensation_seeking": 1.0
            })
        
        batch_request = {"predictions": predictions}
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.predictor')
    def test_model_info_endpoint(self, mock_predictor_attr, client, mock_predictor):
        """Test model info endpoint."""
        mock_predictor_attr.return_value = mock_predictor
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "feature_count" in data
        assert "feature_names" in data
        assert "target_drug" in data
        assert "prediction_type" in data


class TestAPIModels:
    """Test cases for API models."""
    
    def test_prediction_request_model(self, sample_prediction_request):
        """Test PredictionRequest model validation."""
        request = PredictionRequest(**sample_prediction_request)
        
        assert request.age == 25
        assert request.gender == GenderEnum.F
        assert request.education == EducationEnum.UNIVERSITY_DEGREE
        assert request.country == CountryEnum.USA
        assert request.ethnicity == EthnicityEnum.WHITE
        assert request.nscore == 0.5
    
    def test_prediction_request_invalid_age(self):
        """Test PredictionRequest with invalid age."""
        invalid_data = {
            "age": 15,  # Too young
            "gender": "F",
            "education": "University degree",
            "country": "USA",
            "ethnicity": "White",
            "nscore": 0.5,
            "escore": -0.2,
            "oscore": 1.2,
            "ascore": 0.8,
            "cscore": -0.1,
            "impulsiveness": 0.3,
            "sensation_seeking": 1.0
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)
    
    def test_prediction_request_invalid_score(self):
        """Test PredictionRequest with invalid score."""
        invalid_data = {
            "age": 25,
            "gender": "F",
            "education": "University degree",
            "country": "USA",
            "ethnicity": "White",
            "nscore": 5.0,  # Too high
            "escore": -0.2,
            "oscore": 1.2,
            "ascore": 0.8,
            "cscore": -0.1,
            "impulsiveness": 0.3,
            "sensation_seeking": 1.0
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)


class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    @patch('api.main.predictor')
    def test_predict_endpoint_model_not_loaded(self, mock_predictor_attr, client, sample_prediction_request):
        """Test prediction endpoint when model is not loaded."""
        mock_predictor_attr.return_value = None
        
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 503  # Service unavailable
    
    @patch('api.main.predictor')
    def test_predict_endpoint_prediction_failure(self, mock_predictor_attr, client, sample_prediction_request):
        """Test prediction endpoint when prediction fails."""
        mock_predictor = MagicMock()
        mock_predictor.model = MagicMock()
        mock_predictor.predict.return_value = None  # Simulate prediction failure
        mock_predictor_attr.return_value = mock_predictor
        
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 500  # Internal server error
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
