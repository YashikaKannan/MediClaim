"""
Medicare Provider Anomaly Risk Engine - Isolation Forest Engine
Anomaly detection using Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import config
from src.utils import logger, setup_logging, print_section

setup_logging(config.LOGGING_LEVEL)

class IsolationForestEngine:
    """Isolation Forest-based anomaly detection engine"""
    
    def __init__(self, contamination="auto", n_estimators=500, random_state=42):
        """
        Initialize Isolation Forest engine
        
        Parameters:
        -----------
        contamination : str or float
            Contamination parameter for Isolation Forest
        n_estimators : int
            Number of trees in the forest
        random_state : int
            Random state for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.trained = False
    
    def train(self, X):
        """
        Train Isolation Forest model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training features
        """
        print_section(f"Training Isolation Forest (contamination={self.contamination})")
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Contamination: {self.contamination}")
        
        # Create model with config parameters
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples="auto",
            max_features=1.0,
            contamination=self.contamination,
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Train model
        self.model.fit(X)
        
        self.trained = True
        logger.info("Isolation Forest model trained successfully")
        
        # Print model info
        print(f"Number of trees: {self.model.n_estimators}")
        print(f"Random state: {self.random_state}")
        print(f"Contamination: {self.contamination}")
    
    def predict(self, X):
        """
        Get anomaly predictions
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to predict
        
        Returns:
        --------
        np.ndarray
            Predictions (1 = normal, -1 = anomaly)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_normalized(self, X):
        """
        Get normalized anomaly predictions (0 = normal, 1 = anomaly)
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to predict
        
        Returns:
        --------
        np.ndarray
            Normalized predictions (0 or 1)
        """
        predictions = self.predict(X)
        # Convert: 1 -> 0 (normal), -1 -> 1 (anomaly)
        return (predictions == -1).astype(int)
    
    def decision_function(self, X):
        """
        Get raw anomaly scores
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to score
        
        Returns:
        --------
        np.ndarray
            Raw anomaly scores (lower = more anomalous)
        """
        if not self.trained:
            raise ValueError("Model must be trained before scoring")
        
        return self.model.decision_function(X)
    
    def score_samples(self, X):
        """
        Get anomaly scores (higher = more anomalous)
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to score
        
        Returns:
        --------
        np.ndarray
            Anomaly scores (higher = more anomalous)
        """
        if not self.trained:
            raise ValueError("Model must be trained before scoring")
        
        # Isolation Forest score_samples returns negative anomaly scores
        # We negate to get positive scores where higher = more anomalous
        return -self.decision_function(X)


class EnsembleIsolationForest:
    """
    Ensemble of multiple Isolation Forest models for stability
    Uses multiple random seeds and aggregates predictions
    """
    
    def __init__(self, contamination="auto", random_seeds=None):
        """
        Initialize ensemble
        
        Parameters:
        -----------
        contamination : str or float
            Contamination parameter
        random_seeds : list
            List of random seeds for ensemble
        """
        self.contamination = contamination
        self.random_seeds = random_seeds or config.RANDOM_SEEDS
        self.models = []
        self.trained = False
    
    def train(self, X):
        """
        Train ensemble of Isolation Forest models
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training features
        """
        print_section(f"Training Ensemble Isolation Forest ({len(self.random_seeds)} models)")
        
        for i, seed in enumerate(self.random_seeds, 1):
            logger.info(f"Training model {i}/{len(self.random_seeds)} (seed={seed})")
            
            engine = IsolationForestEngine(
                contamination=self.contamination,
                n_estimators=config.ISOLATION_FOREST_CONFIG['n_estimators'],
                random_state=seed
            )
            
            engine.train(X)
            self.models.append(engine)
        
        self.trained = True
        logger.info(f"Ensemble trained: {len(self.models)} models")
    
    def predict_ensemble(self, X):
        """
        Get ensemble predictions (majority voting)
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to predict
        
        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = []
        for engine in self.models:
            pred = engine.predict_normalized(X)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = (predictions.mean(axis=0) >= 0.5).astype(int)
        
        return ensemble_pred
    
    def score_ensemble(self, X):
        """
        Get ensemble anomaly scores (median aggregation)
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to score
        
        Returns:
        --------
        np.ndarray
            Aggregated anomaly scores
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before scoring")
        
        scores = []
        for engine in self.models:
            score = engine.score_samples(X)
            # Normalize scores to 0-1 range
            min_score = score.min()
            max_score = score.max()
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = np.zeros_like(score)
            scores.append(normalized_score)
        
        # Aggregate using median
        scores = np.array(scores)
        ensemble_score = np.median(scores, axis=0)
        
        return ensemble_score
    
    def get_model(self, index=0):
        """Get a specific model from ensemble"""
        if index >= len(self.models):
            raise IndexError(f"Model index {index} out of range")
        return self.models[index]
    
    def get_models(self):
        """Get all models"""
        return self.models


def create_isolation_forest_engine(contamination="auto"):
    """Create single Isolation Forest engine"""
    return IsolationForestEngine(
        contamination=contamination,
        n_estimators=config.ISOLATION_FOREST_CONFIG['n_estimators'],
        random_state=config.RANDOM_STATE
    )


def create_ensemble_isolation_forest(contamination="auto", use_ensemble=True):
    """Create Isolation Forest engine (single or ensemble)"""
    if use_ensemble:
        return EnsembleIsolationForest(
            contamination=contamination,
            random_seeds=config.RANDOM_SEEDS
        )
    else:
        return create_isolation_forest_engine(contamination)
