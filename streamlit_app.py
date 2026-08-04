import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# ===== Internationalization =====
TEXTS = {
    "en": {
        "title": "Fire Detection System",
        "subtitle": "Real-time fire detection using YOLOv8 and laptop webcam",
        "settings": "Settings",
        "language": "Language / Bahasa",
        "model_label": "Select Model",
        "model_custom": "Custom Fire Model",
        "model_default": "Default COCO Model",
        "custom_desc": "Model already trained with fire dataset",
        "default_desc": "Model without fire training data",
        "confidence": "Confidence Threshold",
        "camera": "Live Camera Feed",
        "camera_prompt": "Access Laptop Camera",
        "start_camera": "Start Camera",
        "fire_detected": "⚠️ FIRE DETECTED!",
        "safe": "✅ Safe - No fire detected",
        "details": "Detection Details",
        "status": "Status",
        "howto": """**How to Use:**
1. Click "Start Camera" above
2. Allow camera access in browser
3. Point camera at area to monitor
4. System detects automatically""",
        "model_info": "Model Info",
        "path": "Path",
        "classes": "Classes",
        "device": "Device",
        "note": """**Note:** The default model (COCO) is not specifically trained for fire.
For high accuracy, use a custom model trained with fire datasets.
See `TRAINING_GUIDE.md` for training guide.""",
        "test_image": "Test with Image",
        "upload": "Upload an image to test",
        "result_fire": "🔥 FIRE DETECTED!",
        "result_no_fire": "✅ No fire detected",
        "copyright": "© 2026 gohidiori. All rights reserved.",
        "back": "⬅ Back to Detection",
    },
    "id": {
        "title": "Sistem Deteksi Api",
        "subtitle": "Deteksi api real-time menggunakan YOLOv8 dan kamera laptop",
        "settings": "Pengaturan",
        "language": "Bahasa / Language",
        "model_label": "Pilih Model",
        "model_custom": "Custom Fire Model",
        "model_default": "Default COCO Model",
        "custom_desc": "Model sudah dilatih dengan data api",
        "default_desc": "Model tanpa data pelatihan api",
        "confidence": "Ambang Keyakinan",
        "camera": "Tampilan Kamera Live",
        "camera_prompt": "Akses Kamera Laptop",
        "start_camera": "Mulai Kamera",
        "fire_detected": "⚠️ API TERDETEKSI!",
        "safe": "✅ Aman - Tidak ada api terdeteksi",
        "details": "Detail Deteksi",
        "status": "Status",
        "howto": """**Cara Pakai:**
1. Klik "Mulai Kamera" di atas
2. Izinkan akses kamera browser
3. Arahkan kamera ke area yang ingin dipantau
4. Sistem akan deteksi otomatis""",
        "model_info": "Info Model",
        "path": "Path",
        "classes": "Kelas",
        "device": "Perangkat",
        "note": """**Catatan:** Model default (COCO) tidak terlatih khusus untuk api.
Untuk akurasi tinggi, gunakan model custom yang dilatih dengan dataset api.
Lihat `TRAINING_GUIDE.md` untuk cara training.""",
        "test_image": "Test dengan Gambar",
        "upload": "Upload gambar untuk test",
        "result_fire": "🔥 API TERDETEKSI!",
        "result_no_fire": "✅ Tidak ada api",
        "copyright": "© 2026 gohidiori. Hak cipta dilindungi.",
        "back": "⬅ Kembali ke Deteksi",
    }
}

# ===== Fire Theme CSS (white bg + black text) =====
FIRE_CSS = """
<style>
    /* White background */
    .stApp {
        background: #ffffff;
    }

    /* Compact spacing - reduce to fit one screen */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }
    .stMarkdown, [data-testid="stVerticalBlock"] > div {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    h3, h2 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }

    /* All text black */
    html, body, .stApp, .main, p, span, div, label {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] {
        background: #fff3e0;
        border-right: 2px solid #ff8c00;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
    }

    /* Title - fire gradient (accent only) */
    .fire-title {
        background: linear-gradient(90deg, #ff8c00, #ff4500, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        text-align: center;
        font-size: 3.2rem;
        margin-bottom: 0;
    }
    .fire-subtitle {
        color: #d84315;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .flame-anim {
        font-size: 2.5rem;
        display: inline-block;
        animation: flicker 1.5s infinite;
    }
    @keyframes flicker {
        0%   { opacity: 1; transform: scale(1); }
        25%  { opacity: 0.8; transform: scale(1.05); }
        50%  { opacity: 1; transform: scale(0.98); }
        75%  { opacity: 0.9; transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Cards / containers */
    .fire-card {
        background: #fff3e0;
        border: 2px solid #ff8c00;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .fire-card h3 {
        color: #000000;
        margin-top: 0;
    }

    /* Status badges */
    .badge-fire {
        background: linear-gradient(90deg, #ff4500, #ff0000);
        border: 2px solid #ff8c00;
        color: #ffffff;
        font-weight: bold;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        animation: pulse-red 1s infinite;
    }
    .badge-safe {
        background: linear-gradient(90deg, #43a047, #2e7d32);
        border: 2px solid #ff8c00;
        color: #ffffff;
        font-weight: bold;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 10px #ff0000; }
        50% { box-shadow: 0 0 30px #ff4500; }
        100% { box-shadow: 0 0 10px #ff0000; }
    }

    /* Model badges */
    .model-badge {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .model-badge-custom {
        background: #e8f5e9;
        border: 2px solid #2e7d32;
        color: #1b5e20;
    }
    .model-badge-default {
        background: #eeeeee;
        border: 2px solid #9e9e9e;
        color: #000000;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ff8c00, #ff4500) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: 2px solid #ff8c00 !important;
        border-radius: 10px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff4500, #d84315) !important;
        color: #ffffff !important;
    }

    /* Input widgets */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ff8c00 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #fff3e0;
    }

    /* Info boxes */
    .stAlert, div[data-testid="stInfo"], div[data-testid="stWarning"] {
        background-color: #fff3e0 !important;
        border-left-color: #ff8c00 !important;
        color: #000000 !important;
    }

    /* Copyright footer */
    .footer {
        text-align: center;
        color: #000000;
        font-size: 0.9rem;
        padding: 1rem 0;
        border-top: 2px solid #ff8c00;
        margin-top: 2rem;
        background: #fff3e0;
    }
    .footer a {
        color: #d84315;
        text-decoration: none;
        font-weight: bold;
    }

    /* Camera / upload area */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border-color: #ff8c00 !important;
    }
    div[data-testid="stFileUploaderDropzone"] section,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] span {
        color: #000000 !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #ff8c00, #ff4500) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: 1px solid #ff8c00 !important;
    }
    div[data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(90deg, #ff4500, #d84315) !important;
        color: #ffffff !important;
    }
    div[data-testid="stCameraInput"] {
        border-color: #ff8c00 !important;
        background-color: #ffffff !important;
    }

    /* Selectbox (language dropdown) - black text white bg */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ff8c00 !important;
    }
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] [data-baseweb="listbox"],
    div[data-baseweb="popover"] ul {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"] {
        color: #000000 !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #ffccbc !important;
        color: #000000 !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] div {
        color: #ff8c00 !important;
    }

    /* Animated confidence percentage */
    .conf-percent {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff8c00, #ff4500, #d84315, #ff4500, #ff8c00);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 2s linear infinite;
        margin: 0.4rem 0;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .conf-fire {
        text-align: center;
        font-size: 1.4rem;
        animation: flicker 0.8s infinite;
        margin-bottom: 0.3rem;
    }
</style>
"""

def get_text(lang):
    return TEXTS.get(lang, TEXTS["en"])

@st.cache_resource
def load_model(model_path='best.pt'):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_frame(model, frame, conf_threshold=0.4):
    results = model(frame, stream=True, verbose=False, conf=conf_threshold)
    fire_detected = False
    detections = []
    for r in results:
        annotated_frame = r.plot()
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
                fire_keywords = ['fire', 'flame', 'smoke', 'api', 'asap', 'candle', 'lighter']
                if any(kw in class_name.lower() for kw in fire_keywords):
                    fire_detected = True
    return annotated_frame, fire_detected, detections

def main():
    st.markdown(FIRE_CSS, unsafe_allow_html=True)

    # Language selection in sidebar first
    with st.sidebar:
        lang = st.selectbox(
            "🌐",
            ["Bahasa Indonesia", "English"],
            index=0,
            key="lang_selector"
        )
        lang_code = "id" if lang == "Bahasa Indonesia" else "en"
        t = get_text(lang_code)

        st.header(f"🔥 {t['settings']}")

        # Model selection with descriptions
        st.subheader(f"📦 {t['model_label']}")
        model_option = st.radio(
            t["model_label"],
            ["custom", "default"],
            format_func=lambda x: t["model_custom"] if x == "custom" else t["model_default"],
            key=f"model_radio_{lang_code}"
        )

        if model_option == "custom":
            st.markdown(
                f'<div class="model-badge model-badge-custom">✅ {t["custom_desc"]}</div>',
                unsafe_allow_html=True
            )
            model_path = "best.pt"
            if not os.path.exists(model_path):
                st.sidebar.warning("Model custom tidak ditemukan. Gunakan model default.")
                model_path = 'yolov8n.pt'
                model_option = "default"
        else:
            st.markdown(
                f'<div class="model-badge model-badge-default">ℹ️ {t["default_desc"]}</div>',
                unsafe_allow_html=True
            )
            model_path = 'yolov8n.pt'

        # Confidence threshold
        conf_threshold = st.slider(
            f"🎯 {t['confidence']}",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05
        )

        # Animated percentage below slider
        conf_pct = int(conf_threshold * 100)
        st.markdown(
            f'<div class="conf-fire">🔥</div>'
            f'<div class="conf-percent">{conf_pct}%</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Model info
        st.subheader(f"📊 {t['model_info']}")
        model = load_model(model_path)
        if model is None:
            st.error("Gagal memuat model")
            return
        st.write(f"**{t['path']}:** `{model_path}`")
        st.write(f"**{t['classes']}:** {len(model.names)}")
        st.write(f"**{t['device']}:** {'GPU' if model.device.type == 'cuda' else 'CPU'}")

    # Title (compact)
    st.markdown(f'<h1 class="fire-title" style="font-size:2.2rem;"><span class="flame-anim" style="font-size:1.8rem;">🔥</span> {t["title"]} <span class="flame-anim" style="font-size:1.8rem;">🔥</span></h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="fire-subtitle" style="margin-bottom:0.8rem;font-size:1rem;">{t["subtitle"]}</p>', unsafe_allow_html=True)

    # ===== Main Content - single screen layout =====
    col_left, col_right = st.columns([2, 1], gap="medium")

    with col_left:
        st.markdown(f'<h3 style="margin-bottom:0.3rem;">📹 {t["camera"]}</h3>', unsafe_allow_html=True)

        camera_input = st.camera_input(t["camera_prompt"], key=f"fire_camera_{lang_code}")

        if camera_input is not None:
            bytes_data = camera_input.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv2_img is not None:
                annotated_frame, fire_detected, detections = process_frame(
                    model, cv2_img, conf_threshold
                )
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, channels="RGB", use_container_width=True, width=480)

                if fire_detected:
                    st.markdown(f'<div class="badge-fire">🚨 {t["fire_detected"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-safe">🔒 {t["safe"]}</div>', unsafe_allow_html=True)

                if detections:
                    with st.expander(f"📋 {t['details']}"):
                        for det in detections:
                            st.write(f"- **{det['class']}**: {det['confidence']:.2%} confidence")

    with col_right:
        # Image upload test
        st.markdown(f'<h3 style="margin-bottom:0.3rem;">🖼️ {t["test_image"]}</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            t["upload"],
            type=['jpg', 'jpeg', 'png'],
            key=f"file_uploader_{lang_code}"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            annotated_frame, fire_detected, detections = process_frame(
                model, cv2_img, conf_threshold
            )
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, channels="RGB", caption=t["test_image"], use_container_width=True)

            if fire_detected:
                st.markdown(f'<div class="badge-fire">🚨 {t["result_fire"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-safe">🔒 {t["result_no_fire"]}</div>', unsafe_allow_html=True)

    # ===== Footer =====
    st.markdown(
        f'<div class="footer" style="margin-top:1rem;">🔥 {t["title"]} — © 2026 <b>gohidiori</b>. All rights reserved.</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
