"""
FastAPI application for drug consumption prediction API.
"""

import os
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, ErrorResponse
)
from ..inference import DrugConsumptionPredictor
from ..utils import setup_logging, get_project_root
from ..monitoring import MonitoringService


# Global variables
predictor = None
monitoring_service = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    global predictor, monitoring_service
    
    # Startup
    logging.info("Starting Drug Consumption Prediction API...")
    
    try:
        # Initialize monitoring service
        monitoring_service = MonitoringService()
        logging.info("Monitoring service initialized")
        
        # Initialize predictor
        model_path = os.getenv("MODEL_PATH", str(get_project_root() / "data" / "cannabis_model.pkl"))
        predictor = DrugConsumptionPredictor(model_path)
        
        if predictor.model is None:
            raise RuntimeError("Failed to load model")
            
        logging.info(f"Model loaded successfully from {model_path}")
        
        # Log startup metrics
        monitoring_service.log_startup()
        
    except Exception as e:
        logging.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("Shutting down Drug Consumption Prediction API...")
    if monitoring_service:
        monitoring_service.log_shutdown()


# Create FastAPI application
app = FastAPI(
    title="Drug Consumption Prediction API",
    description="A machine learning API for predicting drug consumption patterns based on personality traits and demographic information.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


def get_predictor() -> DrugConsumptionPredictor:
    """
    Dependency to get the predictor instance.
    
    Returns:
        DrugConsumptionPredictor: The predictor instance
        
    Raises:
        HTTPException: If predictor is not available
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def get_monitoring_service() -> MonitoringService:
    """
    Dependency to get the monitoring service instance.
    
    Returns:
        MonitoringService: The monitoring service instance
    """
    return monitoring_service


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dict[str, str]: API information
    """
    return {
        "message": "Drug Consumption Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Health status information
    """
    global predictor, start_time
    
    model_loaded = predictor is not None and predictor.model is not None
    model_version = "1.0.0" if model_loaded else "unknown"
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=time.time() - start_time
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor: DrugConsumptionPredictor = Depends(get_predictor),
    monitoring: MonitoringService = Depends(get_monitoring_service),
    background_tasks: BackgroundTasks = None
):
    """
    Predict drug consumption for a single sample.
    
    Args:
        request: Prediction request data
        predictor: Predictor dependency
        monitoring: Monitoring service dependency
        background_tasks: Background tasks for logging
        
    Returns:
        PredictionResponse: Prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Convert request to DataFrame
        data_dict = request.dict()
        df = pd.DataFrame([data_dict])
        
        # Make prediction
        prediction = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        if prediction is None or probabilities is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Prepare response
        pred_value = int(prediction.iloc[0])
        pred_label = "High Risk" if pred_value == 1 else "Low Risk"
        prob_dict = probabilities.iloc[0].to_dict()
        confidence = max(prob_dict.values())
        
        response = PredictionResponse(
            prediction=pred_value,
            prediction_label=pred_label,
            probabilities=prob_dict,
            confidence=confidence,
            model_version="1.0.0",
            prediction_id=prediction_id
        )
        
        # Log prediction metrics
        processing_time = (time.time() - start_time) * 1000
        if background_tasks and monitoring:
            background_tasks.add_task(
                monitoring.log_prediction,
                prediction_id=prediction_id,
                prediction=pred_value,
                confidence=confidence,
                processing_time_ms=processing_time,
                input_data=data_dict
            )
        
        return response
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: DrugConsumptionPredictor = Depends(get_predictor),
    monitoring: MonitoringService = Depends(get_monitoring_service),
    background_tasks: BackgroundTasks = None
):
    """
    Predict drug consumption for multiple samples.
    
    Args:
        request: Batch prediction request data
        predictor: Predictor dependency
        monitoring: Monitoring service dependency
        background_tasks: Background tasks for logging
        
    Returns:
        BatchPredictionResponse: Batch prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    start_time = time.time()
    
    try:
        # Convert requests to DataFrame
        data_list = [req.dict() for req in request.predictions]
        df = pd.DataFrame(data_list)
        
        # Make predictions
        predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        if predictions is None or probabilities is None:
            raise HTTPException(status_code=500, detail="Batch prediction failed")
        
        # Prepare responses
        responses = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            pred_value = int(pred)
            pred_label = "High Risk" if pred_value == 1 else "Low Risk"
            prob_dict = prob.to_dict()
            confidence = max(prob_dict.values())
            
            response = PredictionResponse(
                prediction=pred_value,
                prediction_label=pred_label,
                probabilities=prob_dict,
                confidence=confidence,
                model_version="1.0.0",
                prediction_id=str(uuid.uuid4())
            )
            responses.append(response)
        
        # Log batch prediction metrics
        processing_time = (time.time() - start_time) * 1000
        if background_tasks and monitoring:
            background_tasks.add_task(
                monitoring.log_batch_prediction,
                total_predictions=len(responses),
                processing_time_ms=processing_time
            )
        
        return BatchPredictionResponse(
            predictions=responses,
            total_predictions=len(responses),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info(predictor: DrugConsumptionPredictor = Depends(get_predictor)):
    """
    Get information about the loaded model.
    
    Args:
        predictor: Predictor dependency
        
    Returns:
        Dict[str, Any]: Model information
    """
    try:
        feature_names = predictor.get_feature_names()
        return {
            "model_version": "1.0.0",
            "model_type": "RandomForestClassifier",
            "feature_count": len(feature_names) if feature_names else 0,
            "feature_names": feature_names,
            "target_drug": "Cannabis",
            "prediction_type": "Binary Classification"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler.
    
    Args:
        request: Request object
        exc: HTTP exception
        
    Returns:
        JSONResponse: Error response
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler.
    
    Args:
        request: Request object
        exc: Exception
        
    Returns:
        JSONResponse: Error response
    """
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
