import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = ["project_type", "state", "district", "current_stage", "notification_status"]

NUMERICAL_FEATURES = [
    "land_area_acres",
    "affected_families",
    "planned_duration_days",
    "current_elapsed_days",
    "documentation_completion_pct",
    "approval_delay_days",
    "pending_approvals",
    "compensation_approved_pct",
    "compensation_paid_pct",
    "compensation_pending_families",
    "legal_disputes",
    "average_dispute_age_days",
    "ownership_conflicts",
    "rr_progress_pct",
    "rehabilitation_pending_families",
    "possession_pct",
    "stakeholder_response_days",
    "department_backlog",
    "historical_district_delay_rate",
    "historical_project_type_delay_rate"
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
