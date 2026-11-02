"""
AI/Machine Learning Severity Assessment Engine.
Trains and evaluates supervised Random Forest models for automated disaster
severity classification and continuous damage score regression.
Follows the AI enhancement specifications in Section 5.8 of the synopsis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DISASTER_TYPES = ["Flood", "Fire / Chemical Hazard", "Structural Collapse", "Medical Outbreak", "Severe Storm"]
SEVERITY_CLASSES = ["Low", "Medium", "High", "Critical"]

# Label encoder for disaster types
type_encoder = LabelEncoder()
type_encoder.fit(DISASTER_TYPES)


def generate_synthetic_training_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic multi-attribute disaster incident historical training records
    based on NDMA disaster impact indicators.
    """
    np.random.seed(random_state)
    
    types = np.random.choice(DISASTER_TYPES, size=n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    
    # Population impact distribution (skewed log-normal)
    populations = np.random.exponential(scale=350, size=n_samples).astype(int) + 10
    
    # Infrastructure damage index: 1 (cosmetic) to 10 (total collapse)
    damage_indices = np.random.randint(1, 11, size=n_samples)
    
    # Weather severity index: 1 (calm) to 5 (extreme gale/monsoon)
    weather_risks = np.random.randint(1, 6, size=n_samples)
    
    # Locality vulnerability index: 1 (planned sector) to 5 (high-density informal cluster)
    vulnerability_indices = np.random.randint(1, 6, size=n_samples)
    
    # Structural collapse risk flag (1 if buildings compromised, else 0)
    collapse_flags = []
    for t, d in zip(types, damage_indices):
        if t in ["Structural Collapse", "Earthquake"] or d >= 8:
            collapse_flags.append(np.random.choice([0, 1], p=[0.2, 0.8]))
        else:
            collapse_flags.append(np.random.choice([0, 1], p=[0.85, 0.15]))
    collapse_flags = np.array(collapse_flags)
    
    # Ground truth formula for training continuous severity score (0 to 100)
    # Combines population weight, damage index, collapse hazard, and weather escalation
    scores = (
        (damage_indices * 4.5) +
        (np.clip(np.log10(populations) * 12.0, 5, 30)) +
        (weather_risks * 3.0) +
        (vulnerability_indices * 2.5) +
        (collapse_flags * 15.0) +
        np.random.normal(0, 3.5, size=n_samples)
    )
    scores = np.clip(np.round(scores, 1), 5.0, 100.0)
    
    # Discretize into severity classes
    classes = []
    for s in scores:
        if s >= 75.0:
            classes.append("Critical")
        elif s >= 55.0:
            classes.append("High")
        elif s >= 35.0:
            classes.append("Medium")
        else:
            classes.append("Low")
            
    df = pd.DataFrame({
        "disaster_type": types,
        "affected_population": populations,
        "damage_index": damage_indices,
        "weather_risk": weather_risks,
        "vulnerability_index": vulnerability_indices,
        "collapse_flag": collapse_flags,
        "severity_score": scores,
        "severity_class": classes
    })
    return df


class DisasterSeverityMLEngine:
    """
    Trained Supervised Learning Engine for disaster severity scoring and classification.
    """
    def __init__(self):
        self.is_trained = False
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.regressor = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.metrics = {}
        self.feature_names = [
            "disaster_type_enc",
            "affected_population",
            "damage_index",
            "weather_risk",
            "vulnerability_index",
            "collapse_flag"
        ]
        self.feature_labels = [
            "Disaster Category",
            "Affected Population",
            "Infrastructure Damage (1-10)",
            "Weather Escalation (1-5)",
            "Area Vulnerability (1-5)",
            "Structural Collapse Hazard"
        ]
        self._train()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        type_enc = type_encoder.transform(df["disaster_type"])
        X = np.column_stack([
            type_enc,
            df["affected_population"].values,
            df["damage_index"].values,
            df["weather_risk"].values,
            df["vulnerability_index"].values,
            df["collapse_flag"].values
        ])
        return X

    def _train(self):
        data = generate_synthetic_training_data(n_samples=800)
        
        # Split train / test
        train_idx = int(len(data) * 0.8)
        train_data = data.iloc[:train_idx]
        test_data = data.iloc[train_idx:]
        
        X_train = self._prepare_features(train_data)
        X_test = self._prepare_features(test_data)
        
        y_class_train = train_data["severity_class"].values
        y_class_test = test_data["severity_class"].values
        
        y_score_train = train_data["severity_score"].values
        y_score_test = test_data["severity_score"].values
        
        # Fit models
        self.classifier.fit(X_train, y_class_train)
        self.regressor.fit(X_train, y_score_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_class_test, y_pred)
        prec = precision_score(y_class_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_class_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_class_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_class_test, y_pred, labels=SEVERITY_CLASSES)
        
        # Regression metrics
        score_preds = self.regressor.predict(X_test)
        mae = float(np.mean(np.abs(score_preds - y_score_test)))
        
        # Feature importances
        importances = self.classifier.feature_importances_
        feature_importance_dict = dict(zip(self.feature_labels, [round(float(v) * 100, 1) for v in importances]))
        
        self.metrics = {
            "accuracy": round(float(acc) * 100, 1),
            "precision": round(float(prec) * 100, 1),
            "recall": round(float(rec) * 100, 1),
            "f1_score": round(float(f1) * 100, 1),
            "mae": round(mae, 2),
            "confusion_matrix": cm.tolist(),
            "feature_importance": feature_importance_dict,
            "classes": SEVERITY_CLASSES
        }
        self.is_trained = True

    def predict(
        self,
        disaster_type: str,
        affected_population: int,
        damage_index: int = 5,
        weather_risk: int = 3,
        vulnerability_index: int = 3,
        collapse_flag: int = 0
    ) -> Dict[str, Any]:
        """
        Run AI inference on a disaster incident to estimate priority class, score,
        and prediction confidence.
        """
        if disaster_type not in DISASTER_TYPES:
            disaster_type = "Flood"
            
        type_enc = type_encoder.transform([disaster_type])[0]
        x = np.array([[
            type_enc,
            affected_population,
            damage_index,
            weather_risk,
            vulnerability_index,
            collapse_flag
        ]])
        
        pred_class = self.classifier.predict(x)[0]
        probs = self.classifier.predict_proba(x)[0]
        confidence = round(float(np.max(probs)) * 100.0, 1)
        
        pred_score = round(float(self.regressor.predict(x)[0]), 1)
        
        # Class probabilities dictionary
        class_probs = {cls: round(float(p) * 100.0, 1) for cls, p in zip(self.classifier.classes_, probs)}
        
        return {
            "predicted_class": pred_class,
            "predicted_score": pred_score,
            "confidence_pct": confidence,
            "class_probabilities": class_probs
        }


# Singleton instance
_ml_engine = None

def get_ml_engine() -> DisasterSeverityMLEngine:
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = DisasterSeverityMLEngine()
    return _ml_engine
