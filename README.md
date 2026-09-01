# LandRisk AI: Predictive Land Acquisition Governance

> **"Predict delays before they become delays."**
> A Smart India Hackathon (SIH) prototype for infrastructure development risk management.

---

## 📌 Problem Statement

Land acquisition is one of the most critical, complex, and time-sensitive phases of infrastructure development. Delays in land acquisition cost billions in project overruns. These delays occur due to:
- Prolonged administrative approvals
- Legal disputes and litigation
- Delayed compensation disbursements
- Incomplete boundary surveys and documentation
- Pending forest/environmental clearances
- Land ownership conflicts
- Rehabilitation & Resettlement (R&R) bottlenecks
- Inter-departmental coordination hurdles

**LandRisk AI** addresses this challenge by shifting infrastructure monitoring from **reactive tracking** to **predictive governance**. Using Machine Learning trained on historical execution data, the platform identifies which ongoing projects are at risk of delay, explains the specific bottlenecks using Explainable AI (SHAP), suggests prescriptive recommendations, and simulates the impact of administrative interventions using a What-If simulator.

---

## 🛠️ Architecture & Core Components

```
landrisk_ai/
│
├── app.py                      # Main Streamlit Frontend Dashboard
├── requirements.txt            # Python Dependencies
│
├── data/
│   ├── historical_projects.csv # 5,000 completed records for training
│   └── current_projects.csv    # 500 active records for monitoring
│
├── models/
│   ├── delay_classifier.pkl    # Trained RandomForest Classifier
│   ├── delay_regressor.pkl     # Trained RandomForest Regressor
│   ├── preprocessor.pkl        # Fitted Preprocessing pipeline
│   └── model_metadata.json     # Performance metrics and training details
│
├── src/
│   ├── data_generator.py       # Synthesizes correlated datasets
│   ├── preprocessing.py        # Pipelines for scaling & One-Hot Encoding
│   ├── train.py                # Train classification & regression models
│   ├── predict.py              # Inference & Stage-Wise risk assessment
│   ├── explain.py              # SHAP / Fallback Feature Attribution
│   ├── recommendations.py      # Prescriptive Recommendation Engine
│   └── simulator.py            # What-If Intervention Simulator
│
└── README.md                   # System Documentation
```

---

## 🧪 Synthetic Dataset & Correlations

Because real government land acquisition logs are highly sensitive, LandRisk AI generates a realistic, correlated synthetic dataset of **5,000 completed projects** and **500 active projects**.

Features include:
- `project_id`, `project_name`, `project_type`, `state`, `district`, `latitude`, `longitude`
- `land_area_acres`, `affected_families`, `planned_duration_days`, `current_elapsed_days`
- `current_stage`, `notification_status`, `documentation_completion_pct`
- `approval_delay_days`, `pending_approvals`
- `compensation_approved_pct`, `compensation_paid_pct`, `compensation_pending_families`
- `legal_disputes`, `average_dispute_age_days`, `ownership_conflicts`
- `rr_progress_pct`, `rehabilitation_pending_families`
- `possession_pct`, `stakeholder_response_days`, `department_backlog`
- `historical_district_delay_rate`, `historical_project_type_delay_rate`

### Causal Correlative Mapping:
The label `delayed` is calculated using a logistic risk index:
- **Risk Multipliers (increases delay probability)**: Higher legal disputes, higher approval delays, lower documentation completion, high backlog, larger numbers of affected families, and higher stakeholder response times.
- **Risk Mitigators (reduces delay probability)**: Progress on compensation disbursements, rehabilitation progress, and possession handovers.

---

## 🧠 Machine Learning Methodology

1. **Preprocessing**: 
   - Categorical variables are One-Hot Encoded (`handle_unknown='ignore'`).
   - Numerical features are standardized using `StandardScaler`.
   - Preprocessing artifacts are saved to `models/preprocessor.pkl` to align data states.
2. **Classification (Delay Risk)**: 
   - A `RandomForestClassifier` predicts whether a project will experience a significant delay (>30 days past schedule).
   - Outputs: `delay_probability` (0–100%) and a `risk_score` (0–100).
   - Risk Categories:
     - `0–24` = **LOW** (🟢)
     - `25–49` = **MODERATE** (🟡)
     - `50–74` = **HIGH** (🟠)
     - `75–100` = **CRITICAL** (🔴)
3. **Regression (Delay Duration)**:
   - A `RandomForestRegressor` estimates the `expected_delay` in days/months for the projects.
4. **Explainable AI (SHAP)**:
   - For any selected project, SHAP (`shap.TreeExplainer`) determines local feature contributions, showing which features increased or reduced the risk and by how much.
   - A robust fallback mathematical attribution (`(value - mean) * importance`) is implemented to guarantee 100% startup reliability if SHAP compilation fails on target platforms.

---

## 🎮 Demo Guide: Step-by-Step Hackathon Walkthrough

Demonstrate the full intelligence loop to the judges in under 2 minutes:

### Step 1: Open the Dashboard
- Show active projects, total acreage, and families affected.
- Sort the project monitor by **Delay Prob** to identify critical projects.
- Point out the **Model Status: ONLINE** indicator.

### Step 2: Select a Critical Project
- Go to the **Project Risk Details** tab and select a high-risk project (e.g., risk probability >75%).
- Show the **Lifecycle Risk** chart depicting risk profiles for milestones.
- Walk through the **SHAP Explanation**: "Why is this project at risk?" (e.g., Compensation backlog contributing +24% to risk).

### Step 3: View Prescriptive Recommendations
- Point to the **AI Recommendation Engine**: The system reads the top SHAP drivers and issues direct directives (e.g., "Establish camp offices to expedite compensation").

### Step 4: Run the What-If Simulator
- Select the project in the **What-If Simulator** tab.
- Use sliders to simulate interventions: increase **Compensation Paid** (e.g., 40% → 85%) and reduce **Legal Disputes** (e.g., 8 → 2).
- Show the immediate reduction in simulated delay probability (e.g. Current: 82% → Simulated: 48%).

### Step 5: GIS Mapping & Analytics
- Open the **Risk Map** to show geographic distribution of projects colored by risk.
- Open **Analytics** to show administrative dashboards (approval bottlenecks by state, dispute counts, etc.).

### Step 6: Continuous Learning & Performance
- Open the **Model Performance** tab. Show real evaluation metrics calculated from the test set (AUC, F1, MAE).
- Click **Retrain Model** to show how the system updates when new project completion logs are recorded.

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure Python 3.9+ is installed.

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Navigate to project directory
cd /Users/parveshahameds/Documents/Antigravity/LA

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install libraries
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate Data and Train Models
```bash
# Generate synthetic dataset
python3 src/data_generator.py

# Train models and save outputs
python3 src/train.py
```

### 4. Run Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔮 Future Integration
For deployment, LandRisk AI can be integrated with:
1. **PM-GatiShakti National Master Plan**: Laying GIS project tracks directly onto the master plan mapping.
2. **Bhumi Rashi Portal**: Reading real land notification steps, land areas, and compensation logs.
3. **E-Courts Database**: Scraping pending land litigation files automatically based on survey numbers.
4. **State Treasury Systems**: Verifying treasury payment logs for direct compensation monitoring.
