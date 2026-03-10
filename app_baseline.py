import streamlit as st
from predict import FakeNewsPredictor
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="TruthSeeker AI (Baseline)",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #000000 100%);
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #34495e;
        color: #ffffff;
        border: 1px solid #5d6d7e;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #e74c3c;
        box-shadow: 0 0 10px rgba(231, 76, 60, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
    }

    /* Result Cards */
    .result-card {
        background-color: #34495e;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid #5d6d7e;
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #17202a;
        border-right: 1px solid #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/old-time-camera.png", width=80)
    st.title("TruthSeeker (Legacy)")
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Baseline Model
    This version uses the **Original Model** trained ONLY on:
    - **WELFake Dataset** (Political News)
    
    ### ❌ Known Limitations
    - Fails on **AI-Generated Content**.
    - Fails on **Absurd/Sci-Fi Claims**.
    - High accuracy ONLY on political text.
    """)
    st.markdown("---")
    st.caption("v0.1.0 | Baseline Version")

# --- Caching Model Load ---
@st.cache_resource
def get_predictor():
    model_path = 'baseline_model.bin'
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: '{model_path}' not found in directory!")
        st.stop()
    return FakeNewsPredictor(model_path=model_path)

# --- Main Content ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>📉 Baseline Detector</h1>", unsafe_allow_html=True)
    st.info("ℹ️ You are running the **Baseline Model**. Use this to demonstrate failures (e.g., Aliens).")
    
    # Input Section
    st.markdown("### Paste News Article")
    news_text = st.text_area("Enter text here...", height=250, label_visibility="collapsed", placeholder="Try pasting the 'Aliens' story here to see it fail...")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_btn = st.button("🚀 Analyze (Baseline)")

    # Logic
    if analyze_btn:
        if not news_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            try:
                with st.spinner("🧠 Analyzing with Baseline Logic..."):
                    # Simulate processing time for UX
                    time.sleep(0.8)
                    predictor = get_predictor()
                    
                    # DEBUG: Confirm Model Path
                    # st.toast(f"Loaded Model: {predictor.model_path if hasattr(predictor, 'model_path') else 'baseline_model.bin'}")
                    
                    prediction, confidence = predictor.predict(news_text)
                
                # Display Results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                if prediction == "Real":
                    color = "#00b894"
                    icon = "✅"
                    msg = "Likely Real News"
                else:
                    color = "#ff7675"
                    icon = "🚨"
                    msg = "Likely Fake News"
                
                # Result Card
                st.markdown(f"""
                    <div class="result-card" style="border-left: 5px solid {color};">
                        <h2 style="color: {color}; margin: 0;">{icon} {msg}</h2>
                        <p style="color: #b2bec3; margin-top: 10px;">Confidence Score</p>
                        <h1 style="font-size: 48px; margin: 0;">{confidence:.1%}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress Bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(confidence)
                
                # Additional Metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Model Reliability", "Low (Baseline)")
                with m2:
                    st.metric("Processing Time", "0.10s")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
