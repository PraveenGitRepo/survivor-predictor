"""
🚢 Titanic Survivor Predictor - Professional Web Application
Clean, modern interface with professional styling
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="🚢 Titanic Survivor Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f3a93;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Input container */
    .input-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
    }
    
    /* Input label styling */
    .input-label {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    /* Result box styling */
    .result-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .result-title {
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .result-survived {
        color: #27ae60;
    }
    
    .result-not-survived {
        color: #e74c3c;
    }
    
    .probability-text {
        font-size: 2.5em;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .probability-bar {
        width: 100%;
        height: 40px;
        background-color: #e0e0e0;
        border-radius: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9em;
    }
    
    /* Factor styling */
    .factor-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-size: 0.95em;
    }
    
    .factor-positive {
        background-color: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    
    .factor-negative {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 4px solid #f5c6cb;
    }
    
    .factor-neutral {
        background-color: #d1ecf1;
        color: #0c5460;
        border-left: 4px solid #17a2b8;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px;
        font-size: 1.1em;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        background-color: #1f3a93;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #162e73;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Card styling */
    .info-card {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f3a93;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================

@st.cache_resource
def load_model_and_encoders():
    """Load trained model and encoders"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('sex_encoder.pkl', 'rb') as f:
            sex_encoder = pickle.load(f)
        with open('embarked_encoder.pkl', 'rb') as f:
            embarked_encoder = pickle.load(f)
        return model, sex_encoder, embarked_encoder, True
    except FileNotFoundError:
        return None, None, None, False

# Load model
model, sex_encoder, embarked_encoder, is_loaded = load_model_and_encoders()

if not is_loaded:
    st.error("❌ Model files not found!")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🚢 Titanic Survivor Predictor</div>
    <div class="header-subtitle">Fill in the details and click Predict.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CREATE TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analysis", "ℹ️ About"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================

with tab1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">Passenger Class</p>', unsafe_allow_html=True)
        pclass = st.selectbox(
            "Select class",
            options=[1, 2, 3],
            format_func=lambda x: f"{['1st', '2nd', '3rd'][x-1]} Class",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown('<p class="input-label">Gender</p>', unsafe_allow_html=True)
        sex = st.selectbox(
            "Select gender",
            options=['male', 'female'],
            format_func=lambda x: 'Male' if x == 'male' else 'Female',
            label_visibility="collapsed"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">Age</p>', unsafe_allow_html=True)
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown('<p class="input-label">Ticket Fare (£)</p>', unsafe_allow_html=True)
        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=512.0,
            value=50.0,
            label_visibility="collapsed"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">Siblings/Spouses</p>', unsafe_allow_html=True)
        sibsp = st.number_input(
            "SibSp",
            min_value=0,
            max_value=8,
            value=0,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown('<p class="input-label">Parents/Children</p>', unsafe_allow_html=True)
        parch = st.number_input(
            "Parch",
            min_value=0,
            max_value=6,
            value=0,
            label_visibility="collapsed"
        )
    
    st.markdown('<p class="input-label">Port of Embarkation</p>', unsafe_allow_html=True)
    embarked = st.selectbox(
        "Port",
        options=['C', 'Q', 'S'],
        format_func=lambda x: f"{x} - {'Cherbourg' if x == 'C' else 'Queenstown' if x == 'Q' else 'Southampton'}",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    if st.button("🎯 Predict", use_container_width=True, type="primary"):
        try:
            # Encode categorical variables
            sex_encoded = sex_encoder.transform([sex])[0]
            embarked_encoded = embarked_encoder.transform([embarked])[0]
            
            # Create feature array
            X = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
            
            # Make prediction
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.prediction_proba = prediction_proba
            st.session_state.show_result = True
            st.session_state.pclass = pclass
            st.session_state.sex = sex
            st.session_state.age = age
            st.session_state.sibsp = sibsp
            st.session_state.parch = parch
            st.session_state.fare = fare
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if st.session_state.get('show_result', False):
        prediction = st.session_state.prediction
        proba = st.session_state.prediction_proba
        
        # Result box
        if prediction == 1:
            result_text = "✅ SURVIVED"
            result_class = "result-survived"
            survival_prob = proba[1] * 100
            color = "#27ae60"
        else:
            result_text = "❌ DID NOT SURVIVE"
            result_class = "result-not-survived"
            survival_prob = proba[0] * 100
            color = "#e74c3c"
        
        st.markdown(f"""
        <div class="result-container">
            <div class="result-title {result_class}">{result_text}</div>
            <div class="probability-text">{survival_prob:.1f}%</div>
            <p style="color: #666; margin-bottom: 1rem;">Probability of survival based on passenger profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability visualization
        st.markdown(f"""
        <div class="probability-bar">
            <div class="probability-fill" style="width: {proba[1]*100}%; background: linear-gradient(90deg, {'#27ae60' if prediction == 1 else '#e74c3c'}, {'#2ecc71' if prediction == 1 else '#ff6b6b'});">
                {proba[1]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key factors
        st.markdown("### 📌 Key Factors")
        
        pclass = st.session_state.pclass
        sex = st.session_state.sex
        age = st.session_state.age
        fare = st.session_state.fare
        sibsp = st.session_state.sibsp
        parch = st.session_state.parch
        
        factors = []
        
        if pclass == 1:
            factors.append(("✓ First class passengers had better survival rates", "positive"))
        elif pclass == 3:
            factors.append(("✗ Third class passengers had lower survival rates", "negative"))
        else:
            factors.append(("• Second class passengers were in the middle", "neutral"))
        
        if sex == 'female':
            factors.append(("✓ Women had evacuation priority ('women and children first')", "positive"))
        else:
            factors.append(("✗ Men had much lower survival rates", "negative"))
        
        if age < 18:
            factors.append(("✓ Younger passengers had better survival chances", "positive"))
        elif age > 60:
            factors.append(("✗ Elderly passengers had lower survival rates", "negative"))
        else:
            factors.append(("• Adult passengers in middle age range", "neutral"))
        
        if fare > 100:
            factors.append(("✓ Higher ticket fare indicates better accommodations and deck location", "positive"))
        elif fare < 20:
            factors.append(("✗ Low ticket fare indicates poor deck accommodations", "negative"))
        else:
            factors.append(("• Mid-range ticket fare", "neutral"))
        
        if sibsp + parch > 0:
            factors.append(("✓ Traveling with family increased survival chances", "positive"))
        else:
            factors.append(("✗ Traveling alone increased risk", "negative"))
        
        for factor_text, factor_type in factors:
            if factor_type == "positive":
                st.markdown(f'<div class="factor-item factor-positive">{factor_text}</div>', unsafe_allow_html=True)
            elif factor_type == "negative":
                st.markdown(f'<div class="factor-item factor-negative">{factor_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="factor-item factor-neutral">{factor_text}</div>', unsafe_allow_html=True)
        
        # Charts
        st.markdown("### 📊 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Survived', 'Did Not Survive'],
                y=[proba[1] * 100, proba[0] * 100],
                marker=dict(color=['#27ae60', '#e74c3c']),
                text=[f"{proba[1] * 100:.1f}%", f"{proba[0] * 100:.1f}%"],
                textposition="auto"
            )
        ])
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: ANALYSIS
# ============================================================================

with tab2:
    st.markdown("### 📊 Historical Titanic Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Survival by Class")
        class_data = pd.DataFrame({
            'Class': ['1st Class', '2nd Class', '3rd Class'],
            'Survival Rate': [62.0, 41.0, 26.0]
        })
        
        fig = px.bar(
            class_data,
            x='Class',
            y='Survival Rate',
            color='Survival Rate',
            color_continuous_scale=['#e74c3c', '#f39c12', '#27ae60'],
            height=300,
            title=None
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Survival Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Survival by Gender")
        gender_data = pd.DataFrame({
            'Gender': ['Female', 'Male'],
            'Survival Rate': [72.0, 19.0]
        })
        
        fig = px.bar(
            gender_data,
            x='Gender',
            y='Survival Rate',
            color='Survival Rate',
            color_continuous_scale=['#e74c3c', '#27ae60'],
            height=300,
            title=None
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Survival Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📈 Key Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Passengers", "2,224")
    
    with col2:
        st.metric("Survivors", "849 (38.2%)")
    
    with col3:
        st.metric("Casualties", "1,375 (61.8%)")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("### 🚢 About This Application")
    
    st.markdown("""
    **Titanic Survivor Predictor** uses machine learning to predict whether a passenger 
    would have survived the Titanic disaster based on historical passenger data.
    
    #### How It Works
    - Analyzes 7 passenger features
    - Compares against historical patterns
    - Provides probability-based predictions
    - Shows influential factors
    
    #### The Titanic Disaster
    - **When**: April 15, 1912
    - **What**: Sank after iceberg collision
    - **Impact**: ~1,500 casualties out of 2,224 passengers
    - **Significance**: Revealed stark social inequalities
    
    #### Key Historical Patterns
    - **Class Mattered**: 1st class had 3x better survival rates than 3rd class
    - **Gender Priority**: Women were evacuated first ("women and children first")
    - **Age Factor**: Children and young people had better chances
    - **Wealth Impact**: Higher ticket fares = better cabin location and access
    """)
    
    st.markdown("---")
    
    st.markdown("#### 📋 Model Features")
    
    features = pd.DataFrame({
        'Feature': [
            'Passenger Class',
            'Gender',
            'Age',
            'Siblings/Spouses',
            'Parents/Children',
            'Ticket Fare',
            'Embarkation Port'
        ],
        'Description': [
            '1st, 2nd, or 3rd Class',
            'Male or Female',
            '0-100 years',
            'Number aboard (0-8)',
            'Number aboard (0-6)',
            'Price in British Pounds',
            'C, Q, or S'
        ],
        'Impact': [
            'Very High',
            'Very High',
            'High',
            'Medium',
            'Medium',
            'High',
            'Low'
        ]
    })
    
    st.dataframe(features, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.warning("""
    ⚠️ **Disclaimer**
    
    - **Educational Purpose Only**: This is for learning and entertainment
    - **Historical Data**: Based on 1912 Titanic passenger records
    - **Probability Not Certainty**: Results show likelihood, not certainties
    - **Social Context**: Reflects social conditions and biases of that era
    - **No Real-World Use**: Should not be used for serious applications
    """)

# Initialize session state
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
