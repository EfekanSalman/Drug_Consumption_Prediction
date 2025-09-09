"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class GenderEnum(str, Enum):
    """Gender enumeration."""
    F = "F"
    M = "M"


class EducationEnum(str, Enum):
    """Education level enumeration."""
    LEFT_SCHOOL_BEFORE_16 = "Left school before 16 years"
    LEFT_SCHOOL_AT_16 = "Left school at 16 years"
    LEFT_SCHOOL_AT_17 = "Left school at 17 years"
    LEFT_SCHOOL_AT_18 = "Left school at 18 years"
    SOME_COLLEGE = "Some college or university, no certificate or degree"
    PROFESSIONAL_CERT = "Professional certificate/diploma"
    UNIVERSITY_DEGREE = "University degree"
    MASTERS_DEGREE = "Masters degree"
    DOCTORATE_DEGREE = "Doctorate degree"


class CountryEnum(str, Enum):
    """Country enumeration."""
    AUSTRALIA = "Australia"
    CANADA = "Canada"
    NEW_ZEALAND = "New Zealand"
    OTHER = "Other"
    REPUBLIC_OF_IRELAND = "Republic of Ireland"
    UK = "UK"
    USA = "USA"


class EthnicityEnum(str, Enum):
    """Ethnicity enumeration."""
    ASIAN = "Asian"
    BLACK = "Black"
    MIXED_BLACK_ASIAN = "Mixed-Black/Asian"
    MIXED_WHITE_ASIAN = "Mixed-White/Asian"
    MIXED_WHITE_BLACK = "Mixed-White/Black"
    OTHER = "Other"
    WHITE = "White"


class PredictionRequest(BaseModel):
    """
    Request model for drug consumption prediction.
    
    Attributes:
        age (int): Age of the person (18-65)
        gender (GenderEnum): Gender of the person
        education (EducationEnum): Education level
        country (CountryEnum): Country of residence
        ethnicity (EthnicityEnum): Ethnicity
        nscore (float): Neuroticism score (-3 to 3)
        escore (float): Extraversion score (-3 to 3)
        oscore (float): Openness score (-3 to 3)
        ascore (float): Agreeableness score (-3 to 3)
        cscore (float): Conscientiousness score (-3 to 3)
        impulsiveness (float): Impulsiveness score (-2 to 2)
        sensation_seeking (float): Sensation seeking score (-2 to 2)
    """
    
    age: int = Field(..., ge=18, le=65, description="Age of the person (18-65)")
    gender: GenderEnum = Field(..., description="Gender of the person")
    education: EducationEnum = Field(..., description="Education level")
    country: CountryEnum = Field(..., description="Country of residence")
    ethnicity: EthnicityEnum = Field(..., description="Ethnicity")
    nscore: float = Field(..., ge=-3, le=3, description="Neuroticism score (-3 to 3)")
    escore: float = Field(..., ge=-3, le=3, description="Extraversion score (-3 to 3)")
    oscore: float = Field(..., ge=-3, le=3, description="Openness score (-3 to 3)")
    ascore: float = Field(..., ge=-3, le=3, description="Agreeableness score (-3 to 3)")
    cscore: float = Field(..., ge=-3, le=3, description="Conscientiousness score (-3 to 3)")
    impulsiveness: float = Field(..., ge=-2, le=2, description="Impulsiveness score (-2 to 2)")
    sensation_seeking: float = Field(..., ge=-2, le=2, description="Sensation seeking score (-2 to 2)")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Response model for drug consumption prediction.
    
    Attributes:
        prediction (int): Binary prediction (0: Low Risk, 1: High Risk)
        prediction_label (str): Human-readable prediction label
        probabilities (Dict[str, float]): Prediction probabilities for each class
        confidence (float): Confidence score (max probability)
        model_version (str): Version of the model used
        prediction_id (str): Unique identifier for this prediction
    """
    
    prediction: int = Field(..., description="Binary prediction (0: Low Risk, 1: High Risk)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probabilities: Dict[str, float] = Field(..., description="Prediction probabilities for each class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    model_version: str = Field(..., description="Version of the model used")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch drug consumption predictions.
    
    Attributes:
        predictions (List[PredictionRequest]): List of prediction requests
    """
    
    predictions: List[PredictionRequest] = Field(..., min_items=1, max_items=100, 
                                               description="List of prediction requests (1-100 items)")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
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
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch drug consumption predictions.
    
    Attributes:
        predictions (List[PredictionResponse]): List of prediction responses
        total_predictions (int): Total number of predictions made
        processing_time_ms (float): Total processing time in milliseconds
    """
    
    predictions: List[PredictionResponse] = Field(..., description="List of prediction responses")
    total_predictions: int = Field(..., description="Total number of predictions made")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status (str): Health status
        timestamp (str): Current timestamp
        model_loaded (bool): Whether the model is loaded
        model_version (str): Version of the loaded model
        uptime_seconds (float): Application uptime in seconds
    """
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_version: str = Field(..., description="Version of the loaded model")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Attributes:
        error (str): Error type
        message (str): Error message
        details (Optional[Dict[str, Any]]): Additional error details
        timestamp (str): Error timestamp
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
