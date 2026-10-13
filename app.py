import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.applications.inception_v3 import preprocess_input as incept_prep
import google.generativeai as genai

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Diabetic Foot Ulcer AI", page_icon="🦶", layout="wide")

# Replace with your Gemini API Key or set st.secrets["GEMINI_API_KEY"]
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

MODELS_DIR = "models"
CLASSES = ['healthy', 'mild', 'moderate', 'severe']
NUM_CLASSES = 4
NUM_CLINICAL = 11
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
L2_REG = regularizers.l2(1e-4)
SEVERITY_COLORS = {'healthy': '#2ecc71', 'mild': '#f1c40f', 'moderate': '#e67e22', 'severe': '#e74c3c'}
MODEL_COLORS = {'ResNet50': '#e74c3c', 'VGG16': '#3498db', 'InceptionV3': '#2ecc71', 'SwinTransformer': '#9b59b6'}

# ==========================================
# MODEL BUILDERS (Rebuilds + loads .weights.h5)
# ==========================================
def build_film_fusion_head(img_feat, clin_input):
    clin_noise = layers.GaussianNoise(0.15)(clin_input)
    clin_dense = layers.Dense(32, activation='relu', kernel_regularizer=L2_REG)(clin_noise)
    gamma = layers.Dense(128, activation='linear', name='film_gamma')(clin_dense)
    beta = layers.Dense(128, activation='linear', name='film_beta')(clin_dense)
    modulated = layers.Multiply(name='film_scaling')([img_feat, gamma])
    modulated = layers.Add(name='film_shifting')([modulated, beta])
    x = layers.BatchNormalization()(modulated)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    return layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='output')(x)

def build_mm_resnet():
    img_in = keras.Input(shape=INPUT_SHAPE, name='image_input')
    clin_in = keras.Input(shape=(NUM_CLINICAL,), name='clinical_input')
    x_prep = layers.Lambda(resnet_prep, name='resnet_prep')(img_in)
    base = ResNet50(weights=None, include_top=False, input_tensor=x_prep)
    img_feat = layers.GlobalAveragePooling2D()(base.output)
    img_feat = layers.BatchNormalization()(img_feat)
    img_feat = layers.Dense(128, activation='relu', kernel_regularizer=L2_REG)(img_feat)
    out = build_film_fusion_head(img_feat, clin_in)
    return keras.Model(inputs=[img_in, clin_in], outputs=out, name='MM_ResNet50')

def build_mm_vgg16():
    img_in = keras.Input(shape=INPUT_SHAPE, name='image_input')
    clin_in = keras.Input(shape=(NUM_CLINICAL,), name='clinical_input')
    x_prep = layers.Lambda(vgg_prep, name='vgg_prep')(img_in)
    base = VGG16(weights=None, include_top=False, input_tensor=x_prep)
    img_feat = layers.GlobalAveragePooling2D()(base.output)
    img_feat = layers.BatchNormalization()(img_feat)
    img_feat = layers.Dense(128, activation='relu', kernel_regularizer=L2_REG)(img_feat)
    out = build_film_fusion_head(img_feat, clin_in)
    return keras.Model(inputs=[img_in, clin_in], outputs=out, name='MM_VGG16')

def build_mm_inception():
    img_in = keras.Input(shape=INPUT_SHAPE, name='image_input')
    clin_in = keras.Input(shape=(NUM_CLINICAL,), name='clinical_input')
    x_prep = layers.Lambda(incept_prep, name='incept_prep')(img_in)
    base = InceptionV3(weights=None, include_top=False, input_tensor=x_prep)
    img_feat = layers.GlobalAveragePooling2D()(base.output)
    img_feat = layers.BatchNormalization()(img_feat)
    img_feat = layers.Dense(128, activation='relu', kernel_regularizer=L2_REG)(img_feat)
    out = build_film_fusion_head(img_feat, clin_in)
    return keras.Model(inputs=[img_in, clin_in], outputs=out, name='MM_InceptionV3')

class PatchEmbed(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=48, **kw):
        super().__init__(**kw)
        self.proj = layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding='valid')
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.embed_dim = embed_dim
    def call(self, x):
        return self.norm(tf.reshape(self.proj(x / 255.0), [tf.shape(x)[0], -1, self.embed_dim]))

class WindowAttention(layers.Layer):
    def __init__(self, dim, num_heads=4, **kw):
        super().__init__(**kw)
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        self.drop = layers.Dropout(0.1)
    def call(self, x):
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        qkv = tf.transpose(tf.reshape(self.qkv(x), [B, N, 3, self.num_heads, self.head_dim]), [2, 0, 3, 1, 4])
        attn = self.drop(tf.nn.softmax(tf.matmul(qkv[0], qkv[1], transpose_b=True) * self.scale, -1))
        return self.proj(tf.reshape(tf.transpose(tf.matmul(attn, qkv[2]), [0, 2, 1, 3]), [B, N, self.dim]))

class SwinBlock(layers.Layer):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, **kw):
        super().__init__(**kw)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = models.Sequential([layers.Dense(int(dim*mlp_ratio), activation='gelu'), layers.Dropout(0.1), layers.Dense(dim)])
    def call(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class Downsample(layers.Layer):
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.l = layers.Dense(dim * 2)
    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        s = tf.cast(tf.sqrt(tf.cast(N, tf.float32)), tf.int32)
        x = tf.reshape(x, [B, s, s, C])[:, 0::2, 0::2, :]
        return self.l(self.norm(tf.reshape(x, [B, -1, C])))

def build_mm_swin():
    img_in = keras.Input(shape=INPUT_SHAPE, name='image_input')
    clin_in = keras.Input(shape=(NUM_CLINICAL,), name='clinical_input')
    x = PatchEmbed(4, 48)(img_in)
    x = SwinBlock(48, 2)(x); x = Downsample(48)(x)
    x = SwinBlock(96, 4)(x); x = Downsample(96)(x)
    x = SwinBlock(192, 4)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    img_feat = layers.GlobalAveragePooling1D()(x)
    img_feat = layers.Dense(128, activation='gelu', kernel_regularizer=L2_REG)(img_feat)
    out = build_film_fusion_head(img_feat, clin_in)
    return keras.Model(inputs=[img_in, clin_in], outputs=out, name='MM_SwinTransformer')

BUILDERS = {
    'ResNet50': build_mm_resnet,
    'VGG16': build_mm_vgg16,
    'InceptionV3': build_mm_inception,
    'SwinTransformer': build_mm_swin
}

@st.cache_resource
def load_all_models():
    loaded = {}
    for name, builder in BUILDERS.items():
        w_path = os.path.join(MODELS_DIR, f"MM_{name}_weights.weights.h5")
        if os.path.exists(w_path):
            try:
                m = builder()
                m([np.zeros((1, *INPUT_SHAPE), np.float32), np.zeros((1, NUM_CLINICAL), np.float32)])
                m.load_weights(w_path)
                loaded[name] = m
            except Exception as e:
                st.warning(f"Error loading {name}: {e}")
    return loaded

# ==========================================
# SEGMENTATION & MORPHOLOGY
# ==========================================
def segment_ulcer(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    
    m1 = cv2.inRange(hsv, np.array([0,30,30], np.uint8), np.array([20,255,200], np.uint8))
    m2 = cv2.inRange(hsv, np.array([160,30,30], np.uint8), np.array([180,255,200], np.uint8))
    at = cv2.threshold(lab[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    cb = cv2.bitwise_or(cv2.bitwise_or(m1, m2), at)
    cl = cv2.morphologyEx(cb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
    fl = ndimage.binary_fill_holes(cl).astype(np.uint8) * 255
    
    nl, li, st, _ = cv2.connectedComponentsWithStats(fl)
    fm = np.zeros_like(fl)
    if nl > 1: fm[li == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])] = 255
    else: fm = fl
        
    _, fm = cv2.threshold(cv2.GaussianBlur(fm, (5,5), 2), 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(fm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return fm, cnts

def get_18_morphologies(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mb = mask > 0
    se = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    return [
        ("Erosion", cv2.erode(mask, se(9))), ("Dilation", cv2.dilate(mask, se(9))),
        ("Opening", cv2.morphologyEx(mask, cv2.MORPH_OPEN, se(7))),
        ("Gradient", cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, se(5))),
        ("TopHat", cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se(21))),
        ("BlackHat", cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se(21))),
        ("Skeleton", morphology.skeletonize(mb).astype(np.uint8)*255),
        ("Canny", cv2.Canny(mask, 50, 150))
    ]

def get_recommendations(severity, hba1c, neuropathy, prev_ulcer):
    recs = []
    if severity == 'severe':
        recs.append("👞 **Footwear:** Total Contact Cast (TCC) or CROW boot. Strict non-weight-bearing.")
    elif severity == 'moderate':
        recs.append("👞 **Footwear:** Custom therapeutic extra-depth shoes with rocker-bottom soles.")
    else:
        recs.append("👞 **Footwear:** Preventive extra-depth diabetic shoes with wide toe box.")
        
    if neuropathy == "Yes":
        recs.append("🦶 **Insoles:** Custom accommodative orthotics (Plastazote) to distribute pressure.")
        recs.append("⚠️ **Warning:** ZERO barefoot walking allowed. High risk of unnoticed injury.")
        
    if hba1c > 8.0:
        recs.append("🧦 **Socks:** Antimicrobial silver-infused seamless diabetic socks.")
        recs.append("⚠️ **Warning:** Poor glycemic control limits wound healing.")
    return recs

# ==========================================
# DASHBOARD UI
# ==========================================
st.title("🏥 Multimodal Diabetic Foot Ulcer AI Dashboard")

models_dict = load_all_models()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    st.header("2. Clinical Parameters")
    age = st.slider("Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    diab_type = st.selectbox("Diabetes Type", ["Type 2", "Type 1"])
    duration = st.number_input("Duration (Years)", value=10)
    hba1c = st.number_input("HbA1c (%)", value=8.5)
    glucose = st.number_input("Blood Glucose (mg/dL)", value=180)
    neuropathy = st.selectbox("Peripheral Neuropathy?", ["No", "Yes"])
    pad = st.selectbox("Peripheral Arterial Disease (PAD)?", ["No", "Yes"])
    prev_ulcer = st.selectbox("Previous Ulcer History?", ["No", "Yes"])
    callus = st.selectbox("Callus Formation?", ["No", "Yes"])
    
    analyze_btn = st.button("🚀 Analyze & Predict", use_container_width=True, type="primary")

with col2:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

if analyze_btn and uploaded_file is not None:
    clin_vec = np.array([[
        age, int(gender=='Male'), int(diab_type=='Type 2'), duration, hba1c, glucose,
        int(neuropathy=='Yes'), int(pad=='Yes'), 0, int(prev_ulcer=='Yes'), int(callus=='Yes')
    ]], dtype=np.float32)

    img_224 = cv2.cvtColor(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32)
    img_tensor = np.expand_dims(img_224, axis=0)

    preds, prob_list = [], []
    with st.spinner("Analyzing with AI Models..."):
        for name, model in models_dict.items():
            prob = model.predict({'image_input': img_tensor, 'clinical_input': clin_vec}, verbose=0)[0]
            prob = prob / prob.sum()
            prob_list.append(prob)
            preds.append({'Model': name, 'Prediction': CLASSES[np.argmax(prob)].upper(), 'Confidence': f"{np.max(prob)*100:.1f}%"})

    ens_prob = np.mean(prob_list, axis=0)
    final_class = CLASSES[np.argmax(ens_prob)]
    final_conf = np.max(ens_prob) * 100

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🔬 Segmentation", "📈 Probabilities", "💬 Chatbot"])

    with tab1:
        st.markdown(f"""
        <div style="background-color: {SEVERITY_COLORS[final_class]}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h2>🏆 Final Diagnosis: {final_class.upper()}</h2>
            <h4>Confidence: {final_conf:.1f}%</h4>
        </div>
        """, unsafe_allow_html=True)
        st.table(pd.DataFrame(preds))
        for r in get_recommendations(final_class, hba1c, neuropathy, prev_ulcer):
            st.info(r)

    with tab2:
        mask, contours = segment_ulcer(img_bgr)
        overlay = img_rgb.copy()
        mc = np.zeros_like(img_rgb); mc[mask > 0] = [255, 50, 50]
        overlay = cv2.addWeighted(overlay, 1, mc, 0.4, 0)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        c1, c2, c3 = st.columns(3)
        c1.image(img_rgb, caption="Original")
        c2.image(mask, caption="Mask")
        c3.image(overlay, caption="Overlay")
        cols = st.columns(4)
        for i, (title, m_img) in enumerate(get_18_morphologies(img_bgr, mask)):
            cols[i%4].image(m_img, caption=title, cmap="gray" if len(m_img.shape)==2 else None)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([c.upper() for c in CLASSES], ens_prob*100, color=list(SEVERITY_COLORS.values()))
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

    with tab4:
        st.subheader("💬 AI Podiatrist Assistant")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    m = genai.GenerativeModel('gemini-pro')
                    res = m.generate_content(f"Patient context: {final_class.upper()} ulcer, Age {age}, HbA1c {hba1c}%. Question: {prompt}")
                    st.markdown(res.text)
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
                except Exception:
                    st.error("Gemini API Key missing or limit reached.")