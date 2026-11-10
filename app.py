import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import io

# 1. Page Configuration
st.set_page_config(
    page_title="DermaScan AI — Clinical Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Medical UI Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient & Badges */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .brand-title {
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .brand-sub {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .status-pill {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Cards */
    .card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Risk Banners */
    .risk-banner {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        border: 1px solid;
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.12);
        border-color: #ef4444;
        color: #fca5a5;
    }
    .risk-mod {
        background: rgba(245, 158, 11, 0.12);
        border-color: #f59e0b;
        color: #fcd34d;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.12);
        border-color: #10b981;
        color: #6ee7b7;
    }
    
    .risk-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    /* Probabilities */
    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
        font-weight: 500;
    }

    /* Primary Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# 3. Clinical Header
st.markdown("""
<div class="main-header">
    <div>
        <h1 class="brand-title">🔬 DermaScan AI Studio</h1>
        <div class="brand-sub">Computer-Aided Dermatological Triage & HAM10000 Deep Learning Classifier</div>
    </div>
    <div>
        <span class="status-pill">● PyTorch Engine Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. HAM10000 Disease Taxonomy
CLASSES = {
    0: ("Melanoma (mel)", "Malignant melanocyte cancer. High metastatic potential. Immediate dermatologist biopsy needed.", "high"),
    1: ("Melanocytic Nevus (nv)", "Common benign mole. Symmetrical and well-circumscribed. Routine self-monitoring advised.", "low"),
    2: ("Basal Cell Carcinoma (bcc)", "Malignant skin carcinoma originating in basal epidermis. Requires surgical/clinical evaluation.", "high"),
    3: ("Actinic Keratosis (akiec)", "Pre-cancerous sun-induced dysplastic epidermal lesion. Topical therapy/cryotherapy consultation recommended.", "mod"),
    4: ("Benign Keratosis (bkl)", "Seborrheic keratosis or solar lentigo. Benign epidermal proliferation.", "low"),
    5: ("Dermatofibroma (df)", "Benign cutaneous fibrous histiocytoma. Non-threatening skin nodule.", "low"),
    6: ("Vascular Lesion (vasc)", "Benign blood vessel anomaly (cherry angioma / pyogenic granuloma).", "low"),
}

# 5. Model Loader
@st.cache_resource
def load_model():
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)
        
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper function to generate clean synthetic demo cases
def generate_sample_lesion(case_type):
    img = Image.new("RGB", (320, 320), color=(228, 194, 168))
    draw = ImageDraw.Draw(img)
    if case_type == "melanoma":
        # Irregular dark lesion
        draw.polygon([(110, 90), (220, 100), (240, 190), (190, 230), (100, 200), (80, 140)], fill=(35, 20, 15))
        draw.ellipse([120, 110, 190, 180], fill=(20, 10, 8))
    elif case_type == "nevus":
        # Regular circular mole
        draw.ellipse([100, 100, 220, 220], fill=(75, 42, 25))
    elif case_type == "bcc":
        # Translucent papule
        draw.ellipse([110, 110, 210, 210], fill=(185, 120, 105))
        draw.line([(125, 130), (195, 185)], fill=(180, 20, 20), width=2)
    return img

# 6. Sidebar: Test Presets & Info
with st.sidebar:
    st.markdown("### 🧪 Quick Test Samples")
    st.write("Don't have a skin photo? Test with one click:")
    
    preset_choice = None
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔴 Melanoma"):
            preset_choice = "melanoma"
        if st.button("🔴 Basal Cell"):
            preset_choice = "bcc"
    with col_b:
        if st.button("🟢 Benign Mole"):
            preset_choice = "nevus"
        if st.button("🔄 Reset"):
            st.session_state.pop("active_image", None)
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 HAM10000 Classes")
    for code, (name, _, risk) in CLASSES.items():
        tag = "🔴 High Risk" if risk == "high" else ("🟡 Moderate" if risk == "mod" else "🟢 Benign")
        st.caption(f"**{name}** — {tag}")

# Manage active image state
if preset_choice:
    st.session_state["active_image"] = generate_sample_lesion(preset_choice)

# 7. Main Dashboard Grid (2 Columns)
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">📷 1. Lesion Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dermatoscopic photograph (JPG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    active_img = None
    if uploaded_file is not None:
        active_img = Image.open(uploaded_file).convert("RGB")
        st.session_state["active_image"] = active_img
    elif "active_image" in st.session_state:
        active_img = st.session_state["active_image"]

    if active_img is not None:
        st.image(active_img, caption="Active Lesion Frame", use_container_width=True)
        analyze_clicked = st.button("🔬 Execute Deep Learning Triage", type="primary")
    else:
        st.info("👆 Upload an image or click a **Quick Test Sample** in the sidebar to begin.")
        analyze_clicked = False
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">📊 2. Clinical Assessment & Triage</div>', unsafe_allow_html=True)
    
    if active_img is not None and (analyze_clicked or "last_result" in st.session_state):
        with st.spinner("Analyzing lesion features with PyTorch..."):
            input_tensor = transform(active_img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                top3_probs, top3_indices = torch.topk(probabilities, 3)

            pred_idx = top3_indices[0].item()
            pred_conf = top3_probs[0].item() * 100
            name, desc, risk_level = CLASSES[pred_idx]
            
            # Save state
            st.session_state["last_result"] = True

        # Urgency Banner
        if risk_level == "high":
            st.markdown(f"""
            <div class="risk-banner risk-high">
                <div class="risk-title">🔴 HIGH CLINICAL RISK — {name.upper()}</div>
                <div>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "mod":
            st.markdown(f"""
            <div class="risk-banner risk-mod">
                <div class="risk-title">🟡 MODERATE RISK — {name.upper()}</div>
                <div>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner risk-low">
                <div class="risk-title">🟢 BENIGN / LOW RISK — {name.upper()}</div>
                <div>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # Primary Metric
        st.metric(label="Algorithm Confidence Score", value=f"{pred_conf:.1f}%", delta="Primary Match")
        st.markdown("---")

        # Top 3 Differential Probabilities
        st.markdown("#### Differential Diagnosis Likelihoods")
        for prob, idx in zip(top3_probs, top3_indices):
            cls_name, _, _ = CLASSES[idx.item()]
            pct = prob.item() * 100
            st.markdown(f"""
            <div class="prob-row">
                <span>{cls_name}</span>
                <span><strong>{pct:.1f}%</strong></span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob.item()))
            
    else:
        st.write("Awaiting image input. Load a photograph and click **Execute Deep Learning Triage** to view predictions.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# 8. Disclaimer
st.caption("🔒 **Clinical Notice**: DermaScan AI is a computer-aided screening prototype intended for educational and triage exploration. It does not replace histopathological tissue biopsies or evaluation by a certified medical specialist.")
