import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = ["project_type", "state", "district"]

NUMERICAL_FEATURES = [
    "land_area_ha",
    "affected_families",
    "pending_approvals",
    "approval_delay_days",
    "documentation_pct",
    "legal_cases",
    "court_stay",
    "compensation_pct",
    "compensation_delay_days",
    "possession_pct",
    "rr_progress_pct",
    "stakeholder_delay_days",
    "historical_delay_rate",
    "planned_days"
]

def get_preprocessing_pipeline():
    """
    Creates and returns the scikit-learn ColumnTransformer pipeline for preprocessing.
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_feature_names(preprocessor, cat_features=CATEGORICAL_FEATURES, num_features=NUMERICAL_FEATURES):
    """
    Extracts the feature names from a fitted preprocessor ColumnTransformer.
    """
    # Numerical features remain the same
    feature_names = list(num_features)
    
    # Get categorical features after OneHotEncoder
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_onehot_features = cat_encoder.get_feature_names_out(cat_features)
        feature_names.extend(cat_onehot_features)
    except KeyError:
        pass
        
    return feature_names
