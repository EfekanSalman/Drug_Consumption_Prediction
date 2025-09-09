"""
Monitoring system for tracking model performance, data drift, and system metrics.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque


@dataclass
class PredictionMetrics:
    """Data class for prediction metrics."""
    prediction_id: str
    timestamp: datetime
    prediction: int
    confidence: float
    processing_time_ms: float
    input_data: Dict[str, Any]


@dataclass
class BatchMetrics:
    """Data class for batch prediction metrics."""
    timestamp: datetime
    total_predictions: int
    processing_time_ms: float
    avg_processing_time_ms: float


@dataclass
class DataDriftMetrics:
    """Data class for data drift metrics."""
    timestamp: datetime
    feature_name: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    drift_score: float
    is_drift_detected: bool


@dataclass
class ModelPerformanceMetrics:
    """Data class for model performance metrics."""
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    prediction_count: int = 0
    avg_confidence: float = 0.0
    avg_processing_time_ms: float = 0.0


class MonitoringService:
    """
    Monitoring service for tracking model performance, data drift, and system metrics.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the monitoring service.
        
        Args:
            storage_path: Path to store monitoring data. If None, uses default path.
        """
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory storage for real-time metrics
        self.predictions = deque(maxlen=10000)  # Keep last 10k predictions
        self.batch_metrics = deque(maxlen=1000)  # Keep last 1k batch metrics
        self.data_drift_metrics = deque(maxlen=1000)  # Keep last 1k drift metrics
        
        # Reference data for drift detection
        self.reference_data = None
        self.reference_stats = {}
        
        # Performance metrics
        self.performance_metrics = ModelPerformanceMetrics(
            timestamp=datetime.utcnow(),
            prediction_count=0,
            avg_confidence=0.0,
            avg_processing_time_ms=0.0
        )
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Load reference data if available
        self._load_reference_data()
        
        logging.info("Monitoring service initialized")
    
    def _load_reference_data(self):
        """Load reference data for drift detection."""
        reference_file = self.storage_path / "reference_data.json"
        if reference_file.exists():
            try:
                with open(reference_file, 'r') as f:
                    data = json.load(f)
                    self.reference_data = pd.DataFrame(data)
                    self._calculate_reference_stats()
                    logging.info("Reference data loaded for drift detection")
            except Exception as e:
                logging.warning(f"Failed to load reference data: {e}")
    
    def _calculate_reference_stats(self):
        """Calculate reference statistics for drift detection."""
        if self.reference_data is None:
            return
        
        self.reference_stats = {}
        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[column] = {
                'mean': float(self.reference_data[column].mean()),
                'std': float(self.reference_data[column].std())
            }
    
    def set_reference_data(self, data: pd.DataFrame):
        """
        Set reference data for drift detection.
        
        Args:
            data: Reference dataset
        """
        with self.lock:
            self.reference_data = data.copy()
            self._calculate_reference_stats()
            
            # Save reference data
            reference_file = self.storage_path / "reference_data.json"
            try:
                data.to_json(reference_file, orient='records')
                logging.info("Reference data saved for drift detection")
            except Exception as e:
                logging.error(f"Failed to save reference data: {e}")
    
    def log_prediction(self, prediction_id: str, prediction: int, confidence: float, 
                      processing_time_ms: float, input_data: Dict[str, Any]):
        """
        Log a single prediction.
        
        Args:
            prediction_id: Unique prediction identifier
            prediction: Prediction value
            confidence: Prediction confidence
            processing_time_ms: Processing time in milliseconds
            input_data: Input data used for prediction
        """
        with self.lock:
            metrics = PredictionMetrics(
                prediction_id=prediction_id,
                timestamp=datetime.utcnow(),
                prediction=prediction,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                input_data=input_data
            )
            
            self.predictions.append(metrics)
            self._update_performance_metrics()
            self._check_data_drift(input_data)
    
    def log_batch_prediction(self, total_predictions: int, processing_time_ms: float):
        """
        Log batch prediction metrics.
        
        Args:
            total_predictions: Total number of predictions in batch
            processing_time_ms: Total processing time in milliseconds
        """
        with self.lock:
            avg_processing_time = processing_time_ms / total_predictions if total_predictions > 0 else 0
            
            metrics = BatchMetrics(
                timestamp=datetime.utcnow(),
                total_predictions=total_predictions,
                processing_time_ms=processing_time_ms,
                avg_processing_time_ms=avg_processing_time
            )
            
            self.batch_metrics.append(metrics)
    
    def _update_performance_metrics(self):
        """Update performance metrics based on recent predictions."""
        if not self.predictions:
            return
        
        recent_predictions = list(self.predictions)[-100:]  # Last 100 predictions
        
        self.performance_metrics = ModelPerformanceMetrics(
            timestamp=datetime.utcnow(),
            prediction_count=len(self.predictions),
            avg_confidence=np.mean([p.confidence for p in recent_predictions]),
            avg_processing_time_ms=np.mean([p.processing_time_ms for p in recent_predictions])
        )
    
    def _check_data_drift(self, input_data: Dict[str, Any]):
        """
        Check for data drift in input data.
        
        Args:
            input_data: Input data to check for drift
        """
        if not self.reference_stats:
            return
        
        current_data = pd.DataFrame([input_data])
        
        for feature, ref_stats in self.reference_stats.items():
            if feature in current_data.columns:
                current_value = current_data[feature].iloc[0]
                current_mean = current_value  # For single sample
                current_std = 0.0  # For single sample
                
                # Calculate drift score (simplified version)
                drift_score = abs(current_mean - ref_stats['mean']) / ref_stats['std'] if ref_stats['std'] > 0 else 0
                is_drift = drift_score > 2.0  # Threshold for drift detection
                
                drift_metrics = DataDriftMetrics(
                    timestamp=datetime.utcnow(),
                    feature_name=feature,
                    reference_mean=ref_stats['mean'],
                    current_mean=current_mean,
                    reference_std=ref_stats['std'],
                    current_std=current_std,
                    drift_score=drift_score,
                    is_drift_detected=is_drift
                )
                
                self.data_drift_metrics.append(drift_metrics)
                
                if is_drift:
                    logging.warning(f"Data drift detected in feature {feature}: drift_score={drift_score:.3f}")
    
    def log_startup(self):
        """Log application startup."""
        logging.info("Application startup logged")
        self._save_metrics()
    
    def log_shutdown(self):
        """Log application shutdown."""
        logging.info("Application shutdown logged")
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            # Save predictions
            predictions_file = self.storage_path / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
            with open(predictions_file, 'w') as f:
                json.dump([asdict(p) for p in self.predictions], f, default=str)
            
            # Save batch metrics
            batch_file = self.storage_path / f"batch_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(batch_file, 'w') as f:
                json.dump([asdict(b) for b in self.batch_metrics], f, default=str)
            
            # Save drift metrics
            drift_file = self.storage_path / f"drift_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(drift_file, 'w') as f:
                json.dump([asdict(d) for d in self.data_drift_metrics], f, default=str)
            
            # Save performance metrics
            performance_file = self.storage_path / f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(performance_file, 'w') as f:
                json.dump(asdict(self.performance_metrics), f, default=str)
                
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing performance summary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            recent_predictions = [p for p in self.predictions if p.timestamp >= cutoff_time]
            recent_batches = [b for b in self.batch_metrics if b.timestamp >= cutoff_time]
            recent_drift = [d for d in self.data_drift_metrics if d.timestamp >= cutoff_time]
            
            return {
                "time_range_hours": hours,
                "total_predictions": len(recent_predictions),
                "total_batches": len(recent_batches),
                "avg_confidence": np.mean([p.confidence for p in recent_predictions]) if recent_predictions else 0,
                "avg_processing_time_ms": np.mean([p.processing_time_ms for p in recent_predictions]) if recent_predictions else 0,
                "drift_alerts": len([d for d in recent_drift if d.is_drift_detected]),
                "prediction_distribution": {
                    "low_risk": len([p for p in recent_predictions if p.prediction == 0]),
                    "high_risk": len([p for p in recent_predictions if p.prediction == 1])
                }
            }
    
    def get_drift_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get data drift alerts for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of drift alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            recent_drift = [d for d in self.data_drift_metrics if d.timestamp >= cutoff_time and d.is_drift_detected]
            return [asdict(d) for d in recent_drift]
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get current health metrics.
        
        Returns:
            Dict containing health metrics
        """
        with self.lock:
            return {
                "status": "healthy",
                "total_predictions": len(self.predictions),
                "avg_confidence": self.performance_metrics.avg_confidence,
                "avg_processing_time_ms": self.performance_metrics.avg_processing_time_ms,
                "recent_drift_alerts": len([d for d in self.data_drift_metrics if d.is_drift_detected]),
                "last_updated": self.performance_metrics.timestamp.isoformat()
            }
