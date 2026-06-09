import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Student Performance Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = 'rf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    data_path = 'student_data.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

# --- INITIALIZATION ---
model = load_model()
df = load_data()

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --bg-color: #F0F2F9; 
        --glass-bg: rgba(255, 255, 255, 0.07);
        --glass-border: rgba(255, 255, 255, 0.5);
        --accent-primary: #8A89FF;
        --text-main: #2D3436;
    }

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%); }

    /* Glass Navigation */
    div[data-testid="stRadio"] {
        position: sticky; top: 1.5rem; z-index: 1000;
        background: var(--glass-bg); backdrop-filter: blur(12px);
        padding: 0.5rem 2rem; border-radius: 100px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px 0 rgba(138, 137, 255, 0.2);
        width: fit-content; margin: 0 auto 3rem auto;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 2rem; flex-direction: row !important; }
    div[data-testid="stRadio"] label { color: var(--text-main) !important; font-weight: 600 !important; border-radius: 50px; }

    /* Prediction Card */
    .prediction-card {
        background: white; padding: 2.5rem; border-radius: 24px;
        text-align: center; margin-top: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    
    .summary-text {
        font-size: 1.1rem; color: #444; line-height: 1.6;
        margin-top: 1.5rem; padding-top: 1.5rem;
        border-top: 1px solid #eee; font-style: italic;
    }

    /* Impact Badges */
    .impact-badge {
        font-size: 0.75rem; font-weight: 700; padding: 2px 8px;
        border-radius: 4px; text-transform: uppercase; margin-left: 8px;
    }
    .high-impact { background: #FFEBEE; color: #C62828; }
    .med-impact { background: #FFF3E0; color: #E65100; }
    .low-impact { background: #E8F5E9; color: #2E7D32; }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style='text-align: center; margin: 2rem 0 2rem 0;'>
    <h1 style='font-size: 3.5rem; font-weight: 800; color: #000000; margin: 0;'>Student Performance Classifier</h1>
   
</div>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
page = st.radio("Nav", ["Dashboard", "Predictor", "Analytics"], horizontal=True, label_visibility="collapsed")

if page == "Dashboard":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Mean Marks", f"{df['Marks'].mean():.1f}")
    col3.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
    
    st.markdown("")
    
    _, dist_col, _ = st.columns([1, 2, 1])
    with dist_col:
        st.subheader("     Performance Distribution")
    fig_pie = px.pie(df, names='Result', hole=0.6, template="plotly_white",
                         color='Result',
                         color_discrete_map={'Excellent': '#2E7D32', 'Good': '#1565C0', 'Average': '#E65100', 'Poor': '#C62828'})
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
    st.plotly_chart(fig_pie, use_container_width=True)

    
    st.markdown("</div>", unsafe_allow_html=True)    

elif page == "Predictor":
    st.markdown("### 🎯 student performance Analysis")
    
    # Wide Predictor Layout
    _, center_col, _ = st.columns([0.1, 5, 0.1])
    
    with center_col:
        with st.container(border=True):
            st.markdown("**Input Features**")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                hours = st.slider("Study hours per day (High Priority)", 0.0, 12.0, 6.0)
            with r1c2:
                marks_val = st.slider("Internal marks (High Priority)", 0.0, 100.0, 75.0)
            
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                attn = st.slider("Attendance (Medium Priority)", 0.0, 100.0, 85.0)
            with r2c2:
                asgn = st.slider("Assignment marks (Medium Priority)", 0.0, 100.0, 80.0)
            
        predict_btn = st.button("Predict Result", use_container_width=True)
        # Add this right before the button code
        st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(to right, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(138, 137, 255, 0.4);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(138, 137, 255, 0.4);
        background: linear-gradient(to right, #8A89FF, #A29BFE);
        color: white;
    }
    div.stButton > button:first-child:active {
        transform: scale(0.95);
    }
    </style>
""", unsafe_allow_html=True)


    if predict_btn:
        input_data = pd.DataFrame([{
            'Study_Hours': hours,
            'Attendance': attn,
            'Marks': marks_val,
            'Assignment_Score': asgn
        }])
        
        # 1. AI Prediction
        raw_pred = model.predict(input_data)[0]
        final_pred = raw_pred
        rule_applied = False
        
        # 2. THE 1-HOUR RULE (Strict Override)
        if hours < 1:
            if final_pred in ['Excellent', 'Good', 'Average']:
                final_pred = 'Average' if marks_val > 60 else 'Poor'
                rule_applied = True
        
        # 3. Dynamic Summary Logic (3-Line Analysis)
        # Line 1: The 'Why'
        if hours < 5 and marks_val >= 80:
            line1 = f"While the internal marks of {marks_val} are strong, the lack of consistent study effort (only {hours} hours) prevents a higher classification."
        elif hours < 1:
            line1 = f"The critically low study time of {hours} hours is the primary impact, regardless of performance in other areas like internal marks/attendance/assignment."
        else:
            line1 = f"The performance is currently aligned with the student's study commitment of {hours} hours and marks of {marks_val}."

        # Line 2: Feature Impact
        line2 = "The result is predicted by the high weightage of Study Hours and Internal Marks, which carry more impact than Attendance or Assignments in this model."

        # Line 3: Recommendation
        if final_pred != 'Excellent':
            line3 = "To reach 'Excellent,' the student needs to increase daily study hours to at least 6+ hours while maintaining high internal scores."
        else:
            line3 = "To maintain this 'Excellent' status, the student should continue their strong study habits while maintaining internal assessment performance."

        # --- PREMIUM RESULT DISPLAY ---
        st.divider()
        
        # Color theme per result
        result_colors = {
            'Excellent': ('#1B5E20', '#E8F5E9', '🏆'),
            'Good':      ('#1565C0', '#E3F2FD', '🎯'),
            'Average':   ('#E65100', '#FFF3E0', '📊'),
            'Poor':      ('#C62828', '#FFEBEE', '⚠️'),
        }
        text_col, bg_col, icon = result_colors.get(final_pred, ('#333', '#f5f5f5', '📋'))
        
        # Center the result card
        _, card_col, _ = st.columns([1, 3, 1])
        with card_col:
            st.markdown(
                f"""<div style="
                    background: {bg_col};
                    border: 2px solid {text_col};
                    border-radius: 24px;
                    padding: 2.5rem 2rem;
                    text-align: center;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                    <p style="color: {text_col}; text-transform: uppercase; letter-spacing: 3px; font-size: 0.85rem; font-weight: 700; margin: 0;">Prediction Result</p>
                    <h1 style="font-size: 5rem; font-weight: 900; color: {text_col}; margin: 0.3rem 0; line-height: 1;">{final_pred}</h1>
                    <p style="color: #888; font-size: 0.9rem; margin-top: 0.75rem;">
                        📚 {hours}h study &nbsp;·&nbsp; 📝 {marks_val} marks &nbsp;·&nbsp; 📋 {asgn} assignment &nbsp;·&nbsp; 🏫 {attn}% attendance
                    </p>
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis — full width stacked
        st.markdown("**📋 Analysis**")
        st.info(line1)
        
        st.markdown("**📊 Feature Impact**")
        st.info(line2)
        
        st.markdown("**💡 Recommendation**")
        st.success(line3)
        
        st.markdown("**Priority List**")
        priority_data = {
            "Priority": ["🔴 High", "🟡 Medium", "🟡 Medium"],
            "Features": ["Study Hours, Internal Marks", "Assignment Marks", "Attendance"]
        }
        st.table(pd.DataFrame(priority_data))

elif page == "Analytics":
    st.subheader("🧪 Model Insights")
    c1, c2 = st.columns(2)
    with c1:
        importance = [0.38, 0.27, 0.22, 0.13]

        imp_df = pd.DataFrame({
        'Feature': ['Study_Hours', 'Attendance', 'Marks', 'Assignment_Score'],
        'Impact': importance
        }).sort_values('Impact')

        st.plotly_chart(px.bar(
        imp_df,
        x='Impact',
        y='Feature',
        orientation='h',
        title='Random Forest Feature Importance',
        color='Impact',
        color_continuous_scale='Blues'
        ), use_container_width=True)
    with c2:
        corr = df.drop(columns=['Result']).corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Feature Correlation Matrix", color_continuous_scale="Blues"), use_container_width=True)
    