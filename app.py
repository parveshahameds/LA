import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime

# Import project source modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.predict import load_ml_artifacts, predict_project_risk, calculate_stage_wise_risk, get_risk_category
from src.explain import explain_prediction
from src.recommendations import generate_recommendations
from src.simulator import simulate_intervention
from src.data_generator import generate_and_save_data
from src.train import train_and_evaluate

# Configure page settings
st.set_page_config(
    page_title="LandRisk AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #111a2e;
        border: 1px solid #1e2d4a;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Alert styling */
    .alert-card {
        background-color: #1c1515;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .alert-card-high {
        background-color: #1c1815;
        border-left: 5px solid #f97316;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for data loading & model predictions
@st.cache_data
def load_datasets():
    if not os.path.exists("data/historical_projects.csv") or not os.path.exists("data/current_projects.csv"):
        # Run data generator if files are missing
        generate_and_save_data()
        
    hist_df = pd.read_csv("data/historical_projects.csv")
    curr_df = pd.read_csv("data/current_projects.csv")
    return hist_df, curr_df

def get_predictions(curr_df, classifier, regressor, preprocessor):
    return predict_project_risk(curr_df, classifier, regressor, preprocessor)

def load_metadata():
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None

# Initialization
try:
    classifier, regressor, preprocessor = load_ml_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning("Models not trained yet or loading failed. Retraining now...")
    generate_and_save_data()
    train_and_evaluate()
    try:
        classifier, regressor, preprocessor = load_ml_artifacts()
        model_loaded = True
    except:
        st.error("Model initialization failed. Please check the setup.")

if model_loaded:
    hist_df, curr_df = load_datasets()
    
    # Store predictions in session state to avoid recalculating on tab switch
    if 'curr_pred' not in st.session_state:
        st.session_state.curr_pred = get_predictions(curr_df, classifier, regressor, preprocessor)
        st.session_state.model_metadata = load_metadata()
    
    curr_pred = st.session_state.curr_pred
    metadata = st.session_state.model_metadata

    # --- SIDEBAR NAV ---
    st.sidebar.markdown("<h1 style='text-align: center; color: #3b82f6;'>LANDRISK AI</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; font-style: italic; color: #94a3b8;'>\"Predict delays before they become delays.\"</p>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Project Risk Details", "Risk Map", "Analytics", "What-If Simulator", "Intervention Alerts", "Model Performance", "About"]
    )
    
    st.sidebar.divider()
    
    # Sidebar Status Panel
    st.sidebar.markdown("### SYSTEM STATUS")
    st.sidebar.markdown("🟢 **MODEL STATUS:** ONLINE")
    st.sidebar.markdown(f"📊 **DATASET:** {len(hist_df):,} historical projects")
    st.sidebar.markdown("🧠 **MODEL:** Random Forest Ensemble")
    st.sidebar.markdown("🔍 **EXPLAINABILITY:** SHAP Engine")
    st.sidebar.markdown("<p style='font-size:0.75rem; color:#94a3b8;'>Prototype dataset — synthetically generated for demonstration.</p>", unsafe_allow_html=True)
    
    # --- PAGE 1: DASHBOARD ---
    if menu == "Dashboard":
        st.title("LANDRISK AI")
        st.subheader("Predictive Land Acquisition Intelligence")
        st.markdown("*From reactive monitoring to predictive governance.*")
        st.divider()
        
        # Top KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate KPI statistics
        active_cnt = len(curr_pred)
        high_risk_cnt = len(curr_pred[curr_pred["risk_category"] == "HIGH"])
        crit_risk_cnt = len(curr_pred[curr_pred["risk_category"] == "CRITICAL"])
        total_land = curr_pred["land_area_acres"].sum()
        total_families = curr_pred["affected_families"].sum()
        
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{active_cnt}</div><div class='metric-label'>Active Projects</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#f97316;'>{high_risk_cnt}</div><div class='metric-label'>High-Risk Projects</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#ef4444;'>{crit_risk_cnt}</div><div class='metric-label'>Critical Projects</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_land:,.0f}</div><div class='metric-label'>Total Acres</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_families:,.0f}</div><div class='metric-label'>Affected Families</div></div>", unsafe_allow_html=True)
            
        st.divider()
        
        # Project Risk Monitor Header & Filters
        st.markdown("### PROJECT RISK MONITOR")
        
        filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
        
        with filt_col1:
            states_opts = ["All"] + sorted(list(curr_pred["state"].unique()))
            selected_state = st.selectbox("Filter State", states_opts)
            
        with filt_col2:
            if selected_state != "All":
                districts_opts = ["All"] + sorted(list(curr_pred[curr_pred["state"] == selected_state]["district"].unique()))
            else:
                districts_opts = ["All"] + sorted(list(curr_pred["district"].unique()))
            selected_district = st.selectbox("Filter District", districts_opts)
            
        with filt_col3:
            types_opts = ["All"] + sorted(list(curr_pred["project_type"].unique()))
            selected_type = st.selectbox("Filter Project Type", types_opts)
            
        with filt_col4:
            cat_opts = ["All", "LOW", "MODERATE", "HIGH", "CRITICAL"]
            selected_cat = st.selectbox("Filter Risk Category", cat_opts)
            
        # Apply Filters
        filtered_df = curr_pred.copy()
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df["state"] == selected_state]
        if selected_district != "All":
            filtered_df = filtered_df[filtered_df["district"] == selected_district]
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["project_type"] == selected_type]
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df["risk_category"] == selected_cat]
            
        # Sort by risk score (descending)
        filtered_df = filtered_df.sort_values("risk_score", ascending=False)
        
        # Render Table
        display_cols = [
            "project_id", "project_name", "project_type", "state", "district",
            "current_stage", "delay_probability", "risk_category", "expected_delay_months"
        ]
        
        # Clean display names
        rename_dict = {
            "project_id": "ID",
            "project_name": "Project Name",
            "project_type": "Type",
            "state": "State",
            "district": "District",
            "current_stage": "Stage",
            "delay_probability": "Delay Prob (%)",
            "risk_category": "Risk",
            "expected_delay_months": "Exp. Delay (months)"
        }
        
        table_df = filtered_df[display_cols].rename(columns=rename_dict)
        
        # Use streamlit's clean dataframe view
        st.dataframe(
            table_df,
            column_config={
                "Delay Prob (%)": st.column_config.ProgressColumn(
                    "Delay Prob (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "Risk": st.column_config.TextColumn(
                    "Risk Level"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
    # --- PAGE 2: PROJECT RISK DETAILS ---
    elif menu == "Project Risk Details":
        st.title("🔎 PROJECT RISK ANALYSIS")
        
        # Select project dropdown
        project_list = sorted(list(curr_pred["project_name"]))
        selected_project_name = st.selectbox("Select Project to Analyze", project_list)
        
        project_row = curr_pred[curr_pred["project_name"] == selected_project_name].iloc[0]
        
        # Overall Summary Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Project Stage", project_row["current_stage"])
        with col2:
            st.metric("Overall Risk Score", f"{project_row['risk_score']}/100", f"{project_row['risk_category']}")
        with col3:
            st.metric("Expected Delay", f"{project_row['expected_delay_months']} months")
        with col4:
            st.metric("District", f"{project_row['district']} ({project_row['state']})")
            
        st.divider()
        
        # Lifecycle Risk vs Timeline
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### 🔄 LIFECYCLE RISK ASSESSMENT")
            st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:-0.5rem;'>Model-estimated risk at each milestone stage based on project features.</p>", unsafe_allow_html=True)
            
            stage_risks = calculate_stage_wise_risk(project_row)
            stages_list = list(stage_risks.keys())
            risks_list = list(stage_risks.values())
            
            # Map stages to index highlights to show current stage
            colors = []
            for s in stages_list:
                if s == project_row["current_stage"]:
                    colors.append("#3b82f6") # blue for current stage
                else:
                    colors.append("#475569") # slate for others
                    
            fig_stages = go.Figure(go.Bar(
                x=risks_list,
                y=stages_list,
                orientation='h',
                marker_color=colors,
                hovertemplate="Stage: %{y}<br>Risk: %{x}%<extra></extra>"
            ))
            
            fig_stages.update_layout(
                xaxis=dict(title="Risk Probability (%)", range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_stages, use_container_width=True)
            
        with col_right:
            st.markdown("### 📊 KEY STATISTICS")
            stat1, stat2 = st.columns(2)
            with stat1:
                st.write("**Land Area:**", f"{project_row['land_area_acres']:.1f} acres")
                st.write("**Affected Families:**", f"{int(project_row['affected_families'])} families")
                st.write("**Planned Duration:**", f"{int(project_row['planned_duration_days'])} days")
                st.write("**Elapsed Days:**", f"{int(project_row['current_elapsed_days'])} days")
            with stat2:
                # Progress meters
                st.write("**Documentation Progress:**")
                st.progress(float(project_row["documentation_completion_pct"] / 100))
                st.write("**Compensation Paid:**")
                st.progress(float(project_row["compensation_paid_pct"] / 100))
                st.write("**R&R Progress:**")
                st.progress(float(project_row["rr_progress_pct"] / 100))
                st.write("**Possession:**")
                st.progress(float(project_row["possession_pct"] / 100))
                
        st.divider()
        
        # Explainable AI & Recommendations
        col_explain, col_recs = st.columns([1, 1])
        
        with col_explain:
            st.markdown("### 🎯 WHY IS THIS PROJECT AT RISK?")
            st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:-0.5rem;'>Local model attributions (SHAP contributions) driving the prediction.</p>", unsafe_allow_html=True)
            
            top_drivers = explain_prediction(project_row, classifier, preprocessor, hist_df)
            
            # Format drivers for plotting
            driver_names = [d["feature"] for d in top_drivers]
            driver_impacts = [d["shap_val"] * 100 for d in top_drivers]
            driver_colors = ["#ef4444" if i > 0 else "#22c55e" for i in driver_impacts]
            
            fig_shap = go.Figure(go.Bar(
                x=driver_impacts,
                y=driver_names,
                orientation='h',
                marker_color=driver_colors,
                hovertemplate="Driver: %{y}<br>Impact: %{x:+.1f}%<extra></extra>"
            ))
            
            fig_shap.update_layout(
                xaxis=dict(title="Contribution to Risk Probability (%)"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
        with col_recs:
            st.markdown("### 🛡️ AI RECOMMENDATION ENGINE")
            recs = generate_recommendations(top_drivers, project_row)
            
            st.markdown("<div style='background-color:#1e293b; padding:1rem; border-radius:6px; border-left:4px solid #3b82f6;'>", unsafe_allow_html=True)
            st.markdown(f"**TOP RECOMMENDED ACTION:**\n\n{recs['top_action']}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("\n**ADDITIONAL ACTIONS:**")
            for act in recs["additional_actions"]:
                st.markdown(f"- {act}")
                
    # --- PAGE 3: RISK MAP ---
    elif menu == "Risk Map":
        st.title("🗺️ GEOGRAPHIC RISK MAP")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-0.5rem;'>Spatial distribution of land acquisition projects, color-coded by prediction risk level.</p>", unsafe_allow_html=True)
        st.divider()
        
        # Coordinates mapping for map filtering
        map_col1, map_col2, map_col3 = st.columns(3)
        with map_col1:
            states_opts = ["All"] + sorted(list(curr_pred["state"].unique()))
            m_state = st.selectbox("State Filter", states_opts, key="m_state")
        with map_col2:
            types_opts = ["All"] + sorted(list(curr_pred["project_type"].unique()))
            m_type = st.selectbox("Project Type Filter", types_opts, key="m_type")
        with map_col3:
            cat_opts = ["All", "LOW", "MODERATE", "HIGH", "CRITICAL"]
            m_cat = st.selectbox("Risk Level Filter", cat_opts, key="m_cat")
            
        # Apply filters
        m_df = curr_pred.copy()
        if m_state != "All":
            m_df = m_df[m_df["state"] == m_state]
        if m_type != "All":
            m_df = m_df[m_df["project_type"] == m_type]
        if m_cat != "All":
            m_df = m_df[m_df["risk_category"] == m_cat]
            
        # Define RGB colors for PyDeck
        # Low = Green, Moderate = Yellow, High = Orange, Critical = Red
        def get_rgb_color(row):
            cat = row["risk_category"]
            if cat == "LOW":
                return [34, 197, 94, 200]
            elif cat == "MODERATE":
                return [234, 179, 8, 200]
            elif cat == "HIGH":
                return [249, 115, 22, 200]
            else:
                return [239, 68, 68, 200]
                
        m_df["color"] = m_df.apply(get_rgb_color, axis=1)
        # Point sizes proportional to land area
        m_df["radius"] = np.clip(m_df["land_area_acres"] * 10, 3000, 25000)
        
        # PyDeck Scatterplot Layer
        scatterplot_layer = pdk.Layer(
            "ScatterplotLayer",
            m_df,
            get_position="[longitude, latitude]",
            get_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_min_pixels=6,
            radius_max_pixels=30,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255, 255],
        )
        
        # Set initial map view based on data center
        avg_lat = m_df["latitude"].mean() if len(m_df) > 0 else 20.0
        avg_lon = m_df["longitude"].mean() if len(m_df) > 0 else 78.0
        
        view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_lon,
            zoom=5,
            pitch=0
        )
        
        r_map = pdk.Deck(
            layers=[scatterplot_layer],
            initial_view_state=view_state,
            tooltip={
                "html": """
                <b>Project:</b> {project_name}<br/>
                <b>District:</b> {district} ({state})<br/>
                <b>Stage:</b> {current_stage}<br/>
                <b>Risk Category:</b> <span style='font-weight:bold;'>{risk_category}</span><br/>
                <b>Delay Probability:</b> {delay_probability}%<br/>
                <b>Expected Delay:</b> {expected_delay_months} months
                """,
                "style": {"backgroundColor": "#1e293b", "color": "white", "borderRadius": "4px", "zIndex": 1000}
            }
        )
        
        st.pydeck_chart(r_map)
        st.markdown("<p style='font-size:0.8rem; font-style:italic; text-align:center;'>Synthetic project locations mapped for prototype visualization.</p>", unsafe_allow_html=True)
        
    # --- PAGE 4: ANALYTICS ---
    elif menu == "Analytics":
        st.title("📈 DISTRICT & STATE PORTFOLIO ANALYTICS")
        st.divider()
        
        # Grid layout for charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("#### State-Wise Average Delay Risk")
            state_avg = curr_pred.groupby("state")["risk_score"].mean().reset_index().sort_values("risk_score", ascending=False)
            fig_state = px.bar(state_avg, x="state", y="risk_score", color="risk_score",
                               color_continuous_scale="Viridis", labels={"risk_score": "Avg Risk Score", "state": "State"})
            fig_state.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
            st.plotly_chart(fig_state, use_container_width=True)
            
        with row1_col2:
            st.markdown("#### Top 10 High-Risk Districts (Count of High/Critical Projects)")
            high_risk_df = curr_pred[curr_pred["risk_category"].isin(["HIGH", "CRITICAL"])]
            if len(high_risk_df) > 0:
                dist_counts = high_risk_df.groupby(["district", "state"]).size().reset_index(name="count").sort_values("count", ascending=False).head(10)
                fig_dist = px.bar(dist_counts, x="district", y="count", color="state", labels={"count": "High-Risk Projects", "district": "District"})
                fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No projects classified as High or Critical Risk at this moment.")
                
        st.divider()
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("#### Average Delay Duration by Project Type")
            type_delays = curr_pred.groupby("project_type")["expected_delay_months"].mean().reset_index().sort_values("expected_delay_months", ascending=False)
            fig_types = px.bar(type_delays, x="project_type", y="expected_delay_months", color="expected_delay_months",
                               color_continuous_scale="Plasma", labels={"expected_delay_months": "Exp. Delay (months)", "project_type": "Project Type"})
            fig_types.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
            st.plotly_chart(fig_types, use_container_width=True)
            
        with row2_col2:
            st.markdown("#### Legal disputes count vs Delay Probability")
            fig_scatter = px.scatter(curr_pred, x="legal_disputes", y="delay_probability", size="land_area_acres", color="risk_category",
                                     color_discrete_map={"LOW":"#22c55e","MODERATE":"#eab308","HIGH":"#f97316","CRITICAL":"#ef4444"},
                                     labels={"legal_disputes": "Legal Disputes Count", "delay_probability": "Delay Probability (%)"},
                                     hover_name="project_name")
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.divider()
        row3_col1, row3_col2 = st.columns(2)
        
        with row3_col1:
            st.markdown("#### Compensation Approved % vs Delay Probability")
            fig_comp = px.scatter(curr_pred, x="compensation_approved_pct", y="delay_probability", size="affected_families", color="risk_category",
                                  color_discrete_map={"LOW":"#22c55e","MODERATE":"#eab308","HIGH":"#f97316","CRITICAL":"#ef4444"},
                                  labels={"compensation_approved_pct": "Compensation Approved %", "delay_probability": "Delay Probability (%)"},
                                  hover_name="project_name")
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with row3_col2:
            st.markdown("#### Administrative Approval Bottlenecks (Avg Delay Days by State)")
            state_bottlenecks = curr_pred.groupby("state")["approval_delay_days"].mean().reset_index()
            fig_bottlenecks = px.bar(state_bottlenecks, x="state", y="approval_delay_days", color="approval_delay_days",
                                     labels={"approval_delay_days": "Avg Approval Delay (days)", "state": "State"})
            fig_bottlenecks.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300)
            st.plotly_chart(fig_bottlenecks, use_container_width=True)
            
    # --- PAGE 5: WHAT-IF SIMULATOR ---
    elif menu == "What-If Simulator":
        st.title("🎛️ WHAT-IF INTERVENTION SIMULATOR")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-0.5rem;'>Simulate how administrative interventions (e.g. paying compensation, settling disputes) alter predicted project risk.</p>", unsafe_allow_html=True)
        st.divider()
        
        # Select project dropdown
        sim_project_list = sorted(list(curr_pred["project_name"]))
        selected_sim_project = st.selectbox("Select Project to Simulate", sim_project_list, key="sim_proj")
        
        project_row = curr_pred[curr_pred["project_name"] == selected_sim_project].iloc[0]
        
        st.markdown("#### Adjust Intervention Sliders")
        
        col_sliders, col_outcome = st.columns([2, 1.2])
        
        with col_sliders:
            # Inputs
            comp_paid = st.slider("Compensation Paid Pct", 0, 100, int(project_row["compensation_paid_pct"]))
            legal_disp = st.slider("Legal Disputes Count", 0, 25, int(project_row["legal_disputes"]))
            approval_del = st.slider("Administrative Approval Delay (Days)", 0, 365, int(project_row["approval_delay_days"]))
            doc_comp = st.slider("Documentation Completion Pct", 0, 100, int(project_row["documentation_completion_pct"]))
            rr_prog = st.slider("R&R Progress Pct", 0, 100, int(project_row["rr_progress_pct"]))
            backlog = st.slider("Department Backlog Count", 0, 150, int(project_row["department_backlog"]))
            
        with col_outcome:
            # Current values
            current_prob = project_row["delay_probability"]
            current_delay = project_row["expected_delay_months"]
            
            # Predict modified outcomes
            modifications = {
                "compensation_paid_pct": float(comp_paid),
                "legal_disputes": int(legal_disp),
                "approval_delay_days": float(approval_del),
                "documentation_completion_pct": float(doc_comp),
                "rr_progress_pct": float(rr_prog),
                "department_backlog": int(backlog)
            }
            
            sim_prob, sim_delay = simulate_intervention(
                project_row, modifications, classifier, regressor, preprocessor
            )
            
            # Formulate displays
            prob_diff = sim_prob - current_prob
            delay_diff = sim_delay - current_delay
            
            # Render visual panel
            st.markdown("<div style='background-color:#1e293b; padding:1.5rem; border-radius:8px;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center;'>SIMULATION SUMMARY</h3>", unsafe_allow_html=True)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.metric("Current Delay Prob", f"{current_prob:.1f}%")
                st.metric("Current Expected Delay", f"{current_delay:.1f} mo")
            with out_col2:
                # Color code risk categories
                sim_cat, sim_emoji = get_risk_category(int(sim_prob))
                st.metric("Simulated Delay Prob", f"{sim_prob:.1f}%", f"{sim_emoji} {sim_cat}", delta_color="off")
                st.metric("Simulated Delay Time", f"{sim_delay:.1f} mo")
                
            st.divider()
            
            # Highlight Risk Change
            if prob_diff < 0:
                st.success(f"📉 **Risk Probability Reduced:** {prob_diff:.1f} percentage points")
            elif prob_diff > 0:
                st.error(f"📈 **Risk Probability Increased:** +{prob_diff:.1f} percentage points")
            else:
                st.info("No changes in risk probability.")
                
            if delay_diff < 0:
                st.success(f"⏳ **Expected Delay Reduced:** {abs(delay_diff):.1f} months")
            elif delay_diff > 0:
                st.error(f"⏳ **Expected Delay Increased:** +{delay_diff:.1f} months")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            st.info("⚠️ *Simulation only — represents model-estimated risk under the selected conditions.*")
            
    # --- PAGE 6: INTERVENTION ALERTS ---
    elif menu == "Intervention Alerts":
        st.title("🚨 CRITICAL INTERVENTION ALERTS")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-0.5rem;'>Automated warning system flagging projects that exceed the critical risk threshold (75%+ Probability of Delay).</p>", unsafe_allow_html=True)
        st.divider()
        
        alerts_df = curr_pred[curr_pred["delay_probability"] >= 75.0].sort_values("delay_probability", ascending=False)
        
        if len(alerts_df) > 0:
            st.error(f"Found {len(alerts_df)} active projects requiring CRITICAL intervention!")
            
            for idx, row in alerts_df.iterrows():
                # Get explanation to find top driver
                top_drivers = explain_prediction(row, classifier, preprocessor, hist_df)
                primary_driver = top_drivers[0]["feature"] if len(top_drivers) > 0 else "Unknown administrative bottleneck"
                driver_value = top_drivers[0]["display_value"] if len(top_drivers) > 0 else ""
                
                recs = generate_recommendations(top_drivers, row)
                top_action = recs["top_action"]
                
                st.markdown(f"""
                <div class='alert-card'>
                    <h4 style='color:#ef4444; margin-top:0;'>🔴 CRITICAL ALERT: {row['project_name']} (ID: {row['project_id']})</h4>
                    <p style='margin-bottom:0.5rem;'><b>Risk Category:</b> CRITICAL | <b>Delay Probability:</b> {row['delay_probability']}% | <b>Expected Delay:</b> {row['expected_delay_months']} months</p>
                    <p style='margin-bottom:0.5rem;'><b>Current Stage:</b> {row['current_stage']} (District: {row['district']}, State: {row['state']})</p>
                    <p style='margin-bottom:0.5rem;'><b>Primary Risk Driver:</b> {primary_driver} ({driver_value})</p>
                    <p style='margin-bottom:0; font-weight:bold;'>Recommended Intervention: {top_action}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No projects currently meet the Critical Risk threshold (>= 75% delay probability).")
            
        st.divider()
        # High Risk Section (50%-74%)
        high_alerts_df = curr_pred[(curr_pred["delay_probability"] >= 50.0) & (curr_pred["delay_probability"] < 75.0)].sort_values("delay_probability", ascending=False)
        
        if len(high_alerts_df) > 0:
            st.markdown("### 🟠 High-Risk Monitoring (50% - 74% Delay Probability)")
            for idx, row in high_alerts_df.iterrows():
                top_drivers = explain_prediction(row, classifier, preprocessor, hist_df)
                primary_driver = top_drivers[0]["feature"] if len(top_drivers) > 0 else "Unknown bottleneck"
                driver_value = top_drivers[0]["display_value"] if len(top_drivers) > 0 else ""
                
                recs = generate_recommendations(top_drivers, row)
                
                st.markdown(f"""
                <div class='alert-card-high'>
                    <h4 style='color:#f97316; margin-top:0;'>🟠 HIGH RISK: {row['project_name']} (ID: {row['project_id']})</h4>
                    <p style='margin-bottom:0.5rem;'><b>Delay Probability:</b> {row['delay_probability']}% | <b>Expected Delay:</b> {row['expected_delay_months']} months | <b>Stage:</b> {row['current_stage']}</p>
                    <p style='margin-bottom:0.5rem;'><b>Primary Risk Driver:</b> {primary_driver} ({driver_value})</p>
                    <p style='margin-bottom:0; font-weight:bold;'>Suggested Action: {recs['top_action']}</p>
                </div>
                """, unsafe_allow_html=True)
                
    # --- PAGE 7: MODEL PERFORMANCE ---
    elif menu == "Model Performance":
        st.title("⚙️ MODEL MANAGEMENT & PERFORMANCE")
        st.divider()
        
        # Display Metrics
        st.markdown("### Current Model Metrics (Test Set Evaluation)")
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.markdown("#### Classifier Performance (Binary Classification)")
            class_metrics = metadata["metrics"]["classification"]
            
            st.metric("Accuracy", f"{class_metrics['accuracy']:.2%}")
            st.metric("Precision", f"{class_metrics['precision']:.2%}")
            st.metric("Recall (Sensitivity)", f"{class_metrics['recall']:.2%}")
            st.metric("F1 Score", f"{class_metrics['f1_score']:.2%}")
            st.metric("ROC-AUC", f"{class_metrics['roc_auc']:.2%}")
            
        with m_col2:
            st.markdown("#### Regressor Performance (Delay Duration)")
            reg_metrics = metadata["metrics"]["regression"]
            
            st.metric("Mean Absolute Error (MAE)", f"{reg_metrics['mae_days']} days")
            st.metric("Root Mean Squared Error (RMSE)", f"{reg_metrics['rmse_days']} days")
            st.metric("R-Squared (R²)", f"{reg_metrics['r2_score']:.4f}")
            
        st.divider()
        
        # Model Learning Simulation
        st.markdown("### 🔄 MODEL LEARNING & RETRAINING")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-0.5rem;'>Simulate how the model trains continuously on newly closed project records to improve accuracy over time.</p>", unsafe_allow_html=True)
        
        st.write("**Model Version:**", metadata.get("model_version", "1.0.0"))
        st.write("**Last Retrained:**", metadata.get("last_trained", "Unknown"))
        st.write("**Historical Projects Used:**", f"{metadata.get('training_dataset_size', 5000):,} records")
        
        retrain_btn = st.button("Retrain Model")
        
        if retrain_btn:
            with st.spinner("Executing model pipeline (regenerating data and retraining)..."):
                start_time = datetime.now()
                # Run generation and training directly
                generate_and_save_data()
                train_and_evaluate()
                
                # Reload artifacts and clear cache
                st.cache_data.clear()
                classifier, regressor, preprocessor = load_ml_artifacts()
                hist_df, curr_df = load_datasets()
                st.session_state.curr_pred = get_predictions(curr_df, classifier, regressor, preprocessor)
                st.session_state.model_metadata = load_metadata()
                
                # Fetch new values
                curr_pred = st.session_state.curr_pred
                metadata = st.session_state.model_metadata
                
                elapsed = (datetime.now() - start_time).total_seconds()
                st.success(f"Model successfully retrained and cached in {elapsed:.2f} seconds!")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
                
        st.markdown("<p style='font-size:0.8rem; font-style:italic; margin-top:2rem;'>Prototype continuous-learning simulation.</p>", unsafe_allow_html=True)
        
    # --- PAGE 8: ABOUT ---
    elif menu == "About":
        st.title("ℹ️ ABOUT LANDRISK AI")
        st.markdown("### Predictive Land Acquisition Intelligence")
        st.divider()
        
        st.markdown("""
        **Problem Statement:**
        Land acquisition is one of the most critical and time-sensitive phases of infrastructure development in India. Projects frequently suffer from delays due to complex administrative clearances, litigation, delayed compensation disbursements, pending boundary reports, rehabilitation issues, and inter-departmental coordination roadblocks. 
        
        **Solution:**
        **LandRisk AI** addresses this challenge by shifting infrastructure monitoring from *reactive tracking* to *predictive governance*. By deploying Machine Learning algorithms trained on historical execution profiles, LandRisk AI evaluates ongoing projects and forecasts the probability and expected duration of delays before they manifest.
        
        **Core Intelligence Loop:**
        1. **Predictive Scoring:** Random Forest Classifier computes raw risk scores (0–100) mapped to four levels (Low, Moderate, High, Critical).
        2. **Explainable AI (XAI):** The SHAP explanation engine extracts local feature attributions, explaining exactly *why* a project is classified as at-risk (e.g., compensation backlog, pending forest approval).
        3. **Prescriptive Recommendations:** The heuristics engine maps risk drivers to administrative instructions (e.g. prioritizing mediation cells for land disputes).
        4. **What-If Simulation:** Decision-makers can simulate policy interventions (e.g., clearing 80% of compensation backlog) and immediately observe updated project risk scores through model re-inference.
        
        **Continuous Learning:**
        As ongoing projects reach closure, their execution parameters are logged back into the historical dataset. The continuous-learning module retraining feature enables the models to learn new correlation signals, dynamically improving forecasting capabilities.
        
        ---
        **Disclaimer:**
        This platform is a **functional prototype** developed for demonstration purposes. The project datasets, locations, and coordinates shown in this application are **synthetically generated** to demonstrate system capabilities under correlated risk patterns.
        """)
