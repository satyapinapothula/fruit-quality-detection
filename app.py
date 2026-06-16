import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="FreshScan AI",
    page_icon="🍎",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background: #F0F4F8 !important;
}

.block-container { padding-top: 2rem !important; max-width: 780px !important; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 40px 30px 36px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 10px 40px rgba(102,126,234,0.3);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.3);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-sub { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin: 0; }
.hero-fruits { font-size: 2rem; margin-bottom: 14px; }

.card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.result-fresh {
    background: linear-gradient(135deg, #d4f5e2, #a8edca);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    border: 2px solid #4CAF50;
    box-shadow: 0 8px 25px rgba(76,175,80,0.2);
}
.result-rotten {
    background: linear-gradient(135deg, #fde8e8, #fabebe);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    border: 2px solid #f44336;
    box-shadow: 0 8px 25px rgba(244,67,54,0.2);
}
.result-icon { font-size: 3.5rem; margin-bottom: 6px; }
.result-quality-fresh { font-size: 1.9rem; font-weight: 800; color: #1b5e20; margin: 0; }
.result-quality-rotten { font-size: 1.9rem; font-weight: 800; color: #7f0000; margin: 0; }
.result-name-fresh { font-size: 1rem; color: #2e7d32; margin: 4px 0 0 0; font-weight: 500; }
.result-name-rotten { font-size: 1rem; color: #c62828; margin: 4px 0 0 0; font-weight: 500; }

.conf-card-fresh {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    margin-top: 14px;
    box-shadow: 0 6px 20px rgba(76,175,80,0.3);
}
.conf-card-rotten {
    background: linear-gradient(135deg, #f44336, #e53935);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    margin-top: 14px;
    box-shadow: 0 6px 20px rgba(244,67,54,0.3);
}
.conf-label { color: rgba(255,255,255,0.8); font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 4px 0; }
.conf-value { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; line-height: 1; }
.conf-sub { color: rgba(255,255,255,0.75); font-size: 0.78rem; margin: 4px 0 0 0; }

.prob-header { font-size: 0.75rem; font-weight: 700; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px; }
.prob-row { margin-bottom: 12px; }
.prob-name { display: flex; justify-content: space-between; font-size: 0.85rem; color: #444; margin-bottom: 5px; font-weight: 500; }
.prob-track { background: #f0f0f0; border-radius: 10px; height: 10px; overflow: hidden; }
.bar-fresh  { background: linear-gradient(90deg,#4CAF50,#81C784); height:100%; border-radius:10px; transition: width 0.5s ease; }
.bar-rotten { background: linear-gradient(90deg,#f44336,#e57373); height:100%; border-radius:10px; transition: width 0.5s ease; }
.bar-other  { background: #bdbdbd; height:100%; border-radius:10px; }

.tip-box {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border-radius: 16px;
    padding: 18px 22px;
    border-left: 5px solid #FF9800;
}
.tip-title { color: #e65100; font-size: 0.78rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 10px 0; }
.tip-item { color: #5d4037; font-size: 0.85rem; margin: 6px 0; }

.stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }
.stat-card { background: white; border-radius: 14px; padding: 14px; text-align: center; box-shadow: 0 3px 12px rgba(0,0,0,0.06); }
.stat-val { font-size: 1.4rem; font-weight: 800; color: #667eea; margin: 0; }
.stat-label { font-size: 0.72rem; color: #999; margin: 2px 0 0 0; font-weight: 500; }

.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.78rem;
    margin-top: 30px;
    padding: 20px 0;
    border-top: 1px solid #e0e0e0;
}
.stWarning { display: none !important; }
div[data-testid="stImage"] img { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['freshapples','freshbanana','freshoranges','rottenapples','rottenbanana','rottenoranges']
CLASS_EMOJIS = {'freshapples':'🍎','freshbanana':'🍌','freshoranges':'🍊','rottenapples':'🍎','rottenbanana':'🍌','rottenoranges':'🍊'}
CLASS_DISPLAY = {'freshapples':'Fresh Apple','freshbanana':'Fresh Banana','freshoranges':'Fresh Orange','rottenapples':'Rotten Apple','rottenbanana':'Rotten Banana','rottenoranges':'Rotten Orange'}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/home/rgukt/best_fruit_model.h5')

def predict(image, model):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (100, 100))
    arr = np.expand_dims(img / 255.0, axis=0)
    return model.predict(arr, verbose=0)[0]

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-fruits">🍎 🍌 🍊</div>
    <div class="hero-badge">✦ POWERED BY DEEP LEARNING</div>
    <div class="hero-title">FreshScan AI</div>
    <div class="hero-sub">Instantly detect fruit freshness with 85%+ accuracy CNN model</div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <p class="stat-val">85%</p>
        <p class="stat-label">Model Accuracy</p>
    </div>
    <div class="stat-card">
        <p class="stat-val">6</p>
        <p class="stat-label">Fruit Classes</p>
    </div>
    <div class="stat-card">
        <p class="stat-val">CNN</p>
        <p class="stat-label">Architecture</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner('Loading AI model...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f'Error loading model: {e}')
        st.stop()

# Upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader('📤  Upload a fruit image', type=['jpg','jpeg','png'])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown('<div class="card" style="padding:12px">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption='Your uploaded image')
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('🔍 Analyzing fruit quality...'):
        probs = predict(image, model)

    idx        = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    label      = CLASS_NAMES[idx]
    is_fresh   = 'fresh' in label
    emoji      = CLASS_EMOJIS[label]
    display    = CLASS_DISPLAY[label]
    css        = 'fresh' if is_fresh else 'rotten'
    quality    = '✅  FRESH' if is_fresh else '⚠️  ROTTEN'

    with col2:
        st.markdown(f"""
        <div class="result-{css}">
            <div class="result-icon">{emoji}</div>
            <p class="result-quality-{css}">{quality}</p>
            <p class="result-name-{css}">{display}</p>
        </div>
        <div class="conf-card-{css}">
            <p class="conf-label">Confidence Level</p>
            <p class="conf-value">{confidence:.1f}%</p>
            <p class="conf-sub">Model is {'highly' if confidence>80 else 'moderately'} confident</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="prob-header">📊  Probability Breakdown</p>', unsafe_allow_html=True)
    for i, name in enumerate(CLASS_NAMES):
        pct   = float(probs[i]) * 100
        is_top = (i == idx)
        bar   = 'fresh' if 'fresh' in name else 'rotten'
        bold  = 'font-weight:700; color:#333;' if is_top else ''
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-name">
                <span style="{bold}">{CLASS_EMOJIS[name]}  {CLASS_DISPLAY[name]}</span>
                <span style="{bold}">{pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="bar-{bar}" style="width:{pct:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
        <p class="tip-title">💡  Tips for Better Accuracy</p>
        <p class="tip-item">📸  Use images with a plain white or light background</p>
        <p class="tip-item">🎯  Ensure the fruit is centered and fully visible</p>
        <p class="tip-item">💡  Good, even lighting improves detection accuracy</p>
        <p class="tip-item">🍎  Supported fruits: Apple, Banana, Orange</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:30px; color:#aaa;">
        <p style="font-size:3rem; margin:0">📂</p>
        <p style="margin:8px 0 0 0; font-size:0.9rem">Upload a fruit image above to get started</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🍎 Built by <strong style="color:#667eea">Pinapothula Satya Sai Durga</strong>
    &nbsp;|&nbsp; B.Tech CSE · RGUKT Ongole &nbsp;|&nbsp;
    TensorFlow · OpenCV · CNN · Streamlit
</div>
""", unsafe_allow_html=True)
