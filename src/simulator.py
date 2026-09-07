import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from predict import predict_project_risk

def simulate_intervention(project_row, modifications, classifier, regressor, preprocessor):
    """
    Takes a single project row (dict or pandas Series), applies the modifications,
    runs model inference, and returns (new_probability, new_expected_delay_months).
    """
    # Convert series or dict to dict
    modified_dict = dict(project_row).copy()
    
    # Apply modifications
    for k, v in modifications.items():
        modified_dict[k] = v
        
    # Maintain schema aliases
    if "compensation_pct" in modifications:
        modified_dict["compensation_paid_pct"] = modifications["compensation_pct"]
    if "legal_cases" in modifications:
        modified_dict["legal_disputes"] = modifications["legal_cases"]
        if modifications["legal_cases"] == 0:
            modified_dict["court_stay"] = 0
    if "documentation_pct" in modifications:
        modified_dict["documentation_completion_pct"] = modifications["documentation_pct"]
            
    # Convert to DataFrame for inference
    df_instance = pd.DataFrame([modified_dict])
    
    # Run prediction
    pred_df = predict_project_risk(df_instance, classifier, regressor, preprocessor)
    
    new_prob = pred_df.loc[0, "delay_probability"]
    new_delay_months = pred_df.loc[0, "expected_delay_months"]
    
    return new_prob, new_delay_months
