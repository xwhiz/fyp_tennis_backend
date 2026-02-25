import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from typing import Literal, Optional, Tuple


class ShotClassifier:
    """SVM-based classifier for forehand/backhand shot detection."""
    
    def __init__(self, model_path: str = "./models/shot_classifier_svm.pkl", scaler_path: str = "./models/shot_classifier_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_model_loaded = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        self.load_checkpoint()
    
    def extract_features(
        self,
        player_position_top: Optional[dict],
        player_position_bottom: Optional[dict],
        ball_position: Optional[dict],
        net_y: float = 0.0
    ) -> np.ndarray:
        """Extract features from player and ball positions.
        
        Args:
            player_position_top: Dict with 'bbox' key containing [x1, y1, x2, y2] or None
            player_position_bottom: Dict with 'bbox' key containing [x1, y1, x2, y2] or None
            ball_position: Dict with 'x' and 'y' keys or [x, y] list or None
            net_y: Y coordinate of the net (for court side determination)
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract player positions
        if player_position_top is not None:
            if isinstance(player_position_top, dict) and 'bbox' in player_position_top:
                bbox = player_position_top['bbox']
                top_center_x = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0.0
                top_center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0.0
            elif isinstance(player_position_top, list) and len(player_position_top) >= 2:
                top_center_x = player_position_top[0]
                top_center_y = player_position_top[1]
            else:
                top_center_x, top_center_y = 0.0, 0.0
        else:
            top_center_x, top_center_y = 0.0, 0.0
        
        if player_position_bottom is not None:
            if isinstance(player_position_bottom, dict) and 'bbox' in player_position_bottom:
                bbox = player_position_bottom['bbox']
                bottom_center_x = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0.0
                bottom_center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0.0
            elif isinstance(player_position_bottom, list) and len(player_position_bottom) >= 2:
                bottom_center_x = player_position_bottom[0]
                bottom_center_y = player_position_bottom[1]
            else:
                bottom_center_x, bottom_center_y = 0.0, 0.0
        else:
            bottom_center_x, bottom_center_y = 0.0, 0.0
        
        # Extract ball position
        if ball_position is not None:
            if isinstance(ball_position, dict):
                ball_x = ball_position.get('x', 0.0)
                ball_y = ball_position.get('y', 0.0)
            elif isinstance(ball_position, list) and len(ball_position) >= 2:
                ball_x = ball_position[0]
                ball_y = ball_position[1]
            else:
                ball_x, ball_y = 0.0, 0.0
        else:
            ball_x, ball_y = 0.0, 0.0
        
        # Build feature vector
        features = [
            top_center_x,           # Top player X position
            top_center_y,           # Top player Y position
            bottom_center_x,        # Bottom player X position
            bottom_center_y,        # Bottom player Y position
            ball_x,                 # Ball X position
            ball_y,                 # Ball Y position
            ball_x - top_center_x,  # Ball relative to top player X
            ball_y - top_center_y,  # Ball relative to top player Y
            ball_x - bottom_center_x, # Ball relative to bottom player X
            ball_y - bottom_center_y, # Ball relative to bottom player Y
            1.0 if top_center_y < net_y else 0.0,  # Top player in top court
            1.0 if bottom_center_y > net_y else 0.0,  # Bottom player in bottom court
            1.0 if ball_y < net_y else 0.0,  # Ball in top court
        ]
        
        return np.array(features, dtype=np.float32)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """Train the SVM model on provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            test_size: Proportion of data to use for testing
        
        Returns:
            Dictionary with training metrics
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Filter out 'unknown' labels for training
        valid_mask = np.array([label != 'unknown' for label in y])
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) == 0:
            raise ValueError("No valid training samples (excluding 'unknown' labels)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=42, stratify=y_valid
        )
        
        # Initialize scaler if not already loaded
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize or retrain model
        if self.model is None:
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_samples': len(X_valid),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
        }
        
        # Save checkpoint
        self.save_checkpoint()
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Literal["forehand", "backhand", "unknown"]:
        """Predict shot type from features.
        
        Args:
            features: Feature vector (n_features,)
        
        Returns:
            Predicted shot type
        """
        if not self.is_trained():
            return "unknown"
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def save_checkpoint(self):
        """Save model and scaler to disk."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
        if self.scaler is not None:
            joblib.dump(self.scaler, self.scaler_path)
        self.is_model_loaded = True
    
    def load_checkpoint(self) -> bool:
        """Load model and scaler from disk.
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_model_loaded = True
                return True
        except Exception as e:
            print(f"[SHOT CLASSIFIER]: Error loading checkpoint: {str(e)}")
            self.is_model_loaded = False
            return False
        
        return False
    
    def is_trained(self) -> bool:
        """Check if model is trained and loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.is_model_loaded and self.model is not None and self.scaler is not None



