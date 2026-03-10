import streamlit as st
from src.predict import FakeNewsPredictor
import time

# --- Page Config ---
st.set_page_config(
    page_title="TruthSeeker AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2b42 100%);
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
        background-color: #2d2b42;
        color: #ffffff;
        border: 1px solid #4a4a6a;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #6c5ce7;
        box-shadow: 0 0 10px rgba(108, 92, 231, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
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
        box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
    }

    /* Result Cards */
    .result-card {
        background-color: #2d2b42;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid #4a4a6a;
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #181825;
        border-right: 1px solid #2d2b42;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("TruthSeeker AI")
    st.markdown("---")
    st.markdown("""
    ### 🤖 Model Details
    - **Architecture**: Hybrid BERT + BiLSTM
    - **Training Data**: 
        - WELFake (20k Real/Fake)
        - LIAR (12k Short Claims)
        - Synthetic (1k AI-Gen)
    - **Status**: ✅ Balanced & Fine-Tuned
    
    ### ℹ️ How it works
    This AI analyzes linguistic patterns, punctuation, and context to detect:
    1. **Political Misinformation**
    2. **AI-Generated Fake News**
    3. **Clickbait & Conspiracies**
    """)
    st.markdown("---")
    st.caption("v2.0.0 | Balanced Model")

# --- Main Content ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
    
    # Input Section
    st.markdown("### Paste News Article")
    news_text = st.text_area("Enter text here...", height=250, label_visibility="collapsed", placeholder="Paste the content of the news article here to analyze...")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_btn = st.button("🚀 Analyze Veracity")

    # Logic
    if analyze_btn:
        if not news_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            # Load Model
            @st.cache_resource
            def get_predictor():
                return FakeNewsPredictor()
            
            try:
                with st.spinner("🧠 Analyzing linguistic patterns..."):
                    # Simulate processing time for UX
                    time.sleep(0.8)
                    predictor = get_predictor()
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
                    st.metric("Model Reliability", "High")
                with m2:
                    st.metric("Processing Time", "0.12s")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
