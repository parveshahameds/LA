# LandRisk AI: Tech Stack & Workflow

This document outlines the core technology stack and the operational workflow of the **LandRisk AI** platform.

---

## 🛠️ Technology Stack

* **Frontend Dashboard**: Streamlit (Python web application framework)
* **Styling**: HTML5 and custom CSS (for UI cards and alert formats)
* **Data Processing**: Pandas and NumPy
* **Machine Learning**: Scikit-Learn (Random Forest Classifier & Regressor, StandardScaler, and OneHotEncoder)
* **Model Explainability**: SHAP (Shapley Additive exPlanations)
* **GIS Map Layer**: PyDeck (interactive mapping)
* **Analytics Visualizations**: Plotly (interactive charts)
* **Model Storage**: Joblib (for model serialization and loading)
* **Data Layer**: Local CSV Files (for project logs)

---

## 🔄 Operational Workflow (Input to Output)

Below is the workflow of how project data travels through the LandRisk AI pipeline, from raw inputs to predictions, explanations, and intervention simulations:

```mermaid
graph TD
    %% Inputs
    Input[Project Input Data<br>- CSV Record / Form Inputs] --> Preprocess
    
    %% Preprocessing
    subgraph Preprocessing
        Preprocess[Pipeline Transformer<br>- Encode categories<br>- Scale numerics]
    end
    
    %% Inference
    Preprocess --> ML_Models
    subgraph ML Inference
        ML_Models[Ensemble Models]
        ML_Models -->|Classifier| Risk[Delay Probability & Score<br>- LOW / MODERATE / HIGH / CRITICAL]
        ML_Models -->|Regressor| Duration[Expected Delay in Months]
    end
    
    %% Explanations & Recs
    Risk --> Explainer
    subgraph Explainability & Prescriptions
        Explainer[SHAP Explainer<br>- Compute local feature impact] --> Drivers[Identify Top Delay Drivers]
        Drivers --> Heuristics[Heuristics Engine<br>- Maps drivers to recommendations]
    end
    
    %% Dashboard Displays
    Duration --> Visualizations
    Risk --> Visualizations
    Heuristics --> Visualizations
    subgraph Output Dashboard
        Visualizations[UI Displays<br>- GIS Risk Map<br>- Risk KPI Cards<br>- Action Recommendations]
    end
    
    %% Simulation Loop
    Visualizations -->|What-If Intervention Sliders| Simulator[Simulator Engine<br>- Overrides variables<br>- Re-runs pipeline]
    Simulator -->|Simulated Changes| Input
```
