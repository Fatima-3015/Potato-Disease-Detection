import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="PotatoCare AI",
    page_icon="🥔",
    layout="wide"
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #1C1208;
}

.stApp {
    background: linear-gradient(135deg, #1C1208 0%, #2D1F0E 50%, #1A2A0E 100%);
    min-height: 100vh;
}

/* ── Header ── */
.hero-box {
    background: linear-gradient(135deg, #3E2010 0%, #4A3010 50%, #2A3A10 100%);
    border: 1px solid #8B6914;
    border-radius: 16px;
    padding: 30px 40px;
    text-align: center;
    margin-bottom: 28px;
}
.hero-title {
    font-family: 'Merriweather', serif;
    font-size: 46px;
    font-weight: 700;
    color: #F5DEB3;
    margin: 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
}
.hero-subtitle-en {
    font-size: 16px;
    color: #C8A96E;
    margin: 6px 0 2px 0;
}
.hero-subtitle-ur {
    font-size: 18px;
    color: #A8D878;
    font-family: 'Open Sans', sans-serif;
    margin: 2px 0;
    direction: rtl;
}
.hero-tagline {
    font-size: 13px;
    color: #8B7355;
    margin-top: 8px;
}

/* ── Upload Card ── */
.upload-card {
    background: linear-gradient(145deg, #2A1A08, #3A2A12);
    border: 2px dashed #8B6914;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-bottom: 20px;
}
.upload-title {
    color: #F5DEB3;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
}
.upload-title-ur {
    color: #A8D878;
    font-size: 16px;
    direction: rtl;
    margin-bottom: 12px;
}

/* ── Step Cards ── */
.step-card {
    background: linear-gradient(145deg, #2A1A08, #3A2A12);
    border: 1px solid #5A4020;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0;
}
.step-header {
    font-size: 15px;
    font-weight: 600;
    color: #F5DEB3;
    margin-bottom: 4px;
}
.step-header-ur {
    font-size: 14px;
    color: #A8D878;
    direction: rtl;
    margin-bottom: 8px;
}
.step-content {
    font-size: 13px;
    color: #C8A96E;
    line-height: 1.6;
}

/* ── Result Boxes ── */
.result-healthy {
    background: linear-gradient(135deg, #1A3A10, #2A4A18);
    border: 2px solid #4CAF50;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.result-early {
    background: linear-gradient(135deg, #3A2A08, #4A3810);
    border: 2px solid #FF9800;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.result-late {
    background: linear-gradient(135deg, #3A0808, #4A1010);
    border: 2px solid #F44336;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.result-uncertain {
    background: linear-gradient(135deg, #2A2808, #3A3810);
    border: 2px solid #FFC107;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.result-title-en {
    font-size: 28px;
    font-weight: 700;
    color: #F5DEB3;
    margin: 0;
}
.result-title-ur {
    font-size: 20px;
    color: #A8D878;
    direction: rtl;
    margin: 4px 0;
}
.result-confidence {
    font-size: 14px;
    color: #C8A96E;
    margin-top: 8px;
}

/* ── Confidence Bar ── */
.conf-bar-bg {
    background: #3A2A12;
    border-radius: 20px;
    height: 22px;
    margin: 10px 0;
    overflow: hidden;
    border: 1px solid #5A4020;
}
.conf-bar-fill {
    height: 22px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    color: white;
}

/* ── Severity Badge ── */
.severity-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin: 8px 4px;
}

/* ── Advice Cards ── */
.advice-card {
    background: #2A1A08;
    border-left: 4px solid #8B6914;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
}
.advice-en {
    font-size: 13px;
    color: #F5DEB3;
    margin-bottom: 3px;
}
.advice-ur {
    font-size: 13px;
    color: #A8D878;
    direction: rtl;
}

/* ── Pipeline ── */
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.pipe-step {
    background: #3E2010;
    border: 1px solid #8B6914;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #F5DEB3;
}
.pipe-arrow {
    color: #8B6914;
    font-size: 16px;
}

/* ── Divider ── */
.earthy-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #8B6914, transparent);
    margin: 20px 0;
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    background: #2A1A08 !important;
    border: 2px dashed #8B6914 !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #5A3A10, #4A6A18) !important;
    color: #F5DEB3 !important;
    border: 1px solid #8B6914 !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7A5A20, #5A8A28) !important;
    border-color: #F5DEB3 !important;
}
.stSpinner > div {
    border-top-color: #8B6914 !important;
}
div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 2px solid #5A4020;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-box'>
    <p class='hero-title'>🥔 PotatoCare AI</p>
    <p class='hero-subtitle-en'>Real-World Potato Disease & Crop Health Assistant</p>
    <p class='hero-subtitle-ur'>آلو کی بیماریوں کی تشخیص اور فصل کی صحت کا نظام</p>
    <p class='hero-tagline'>Powered by Deep Learning &nbsp;|&nbsp; NTU PFAI Project 2026 &nbsp;|&nbsp; Fatima Mahmood · Momina Afzaal · Zemal Fatima</p>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE PREVIEW ──────────────────────────────────────────────────────────
st.markdown("""
<div class='step-card'>
    <div class='step-header'>🔬 Detection Pipeline &nbsp;|&nbsp; <span style='color:#A8D878; direction:rtl;'>تشخیص کا طریقہ کار</span></div>
    <div class='pipeline-row'>
        <span class='pipe-step'>📷 Image Quality</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🌿 Leaf Check</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🧠 Detection</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>📊 Confidence</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🔴 Severity</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>💊 Treatment</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🛡️ Prevention</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='earthy-divider'></div>", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return load_model("potato_disease_model.keras")

model = load_my_model()
classes = ["Early Blight", "Late Blight", "Healthy"]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def check_quality(img):
    w, h = img.size
    if w < 100 or h < 100:
        return False, "Image too small (min 100×100px)", "تصویر بہت چھوٹی ہے"
    gray = np.array(img.convert('L'))
    brightness = np.mean(gray)
    if brightness < 30:
        return False, "Image too dark — use better lighting", "تصویر بہت تاریک ہے — روشنی بہتر کریں"
    if brightness > 245:
        return False, "Image overexposed — avoid direct sunlight", "تصویر بہت روشن ہے"
    return True, "Image quality is good", "تصویر کا معیار ٹھیک ہے"

def check_leaf(img):
    arr = np.array(img.convert('RGB'))
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green = np.sum((g > r) & (g > b) & (g > 40))
    brown = np.sum((r > 80) & (g > 50) & (b < 80) & (r > g))
    total = arr.shape[0] * arr.shape[1]
    green_r = green / total
    brown_r = brown / total
    leaf_r  = green_r + brown_r
    return leaf_r, green_r, brown_r

def get_severity(conf, result, brown_r):
    if result == "Healthy":
        return "None", "#4CAF50"
    if brown_r > 0.4 or conf > 95:
        return "Severe 🔴 | شدید", "#F44336"
    elif brown_r > 0.2 or conf > 85:
        return "Moderate 🟠 | درمیانہ", "#FF9800"
    else:
        return "Mild 🟡 | ہلکا", "#FFC107"

# ── LAYOUT ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("""
    <div class='upload-card'>
        <p class='upload-title'>📁 Upload Potato Leaf Image</p>
        <p class='upload-title-ur'>آلو کے پتے کی تصویر اپ لوڈ کریں</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Now | ابھی تجزیہ کریں")

with right_col:
    if uploaded_file and analyze:

        # ── STEP 1 ────────────────────────────────────────────────────────
        st.markdown("""
        <div class='step-card'>
            <div class='step-header'>📷 Step 1: Image Quality Check</div>
            <div class='step-header-ur'>تصویر کے معیار کی جانچ</div>
        """, unsafe_allow_html=True)

        ok, msg_en, msg_ur = check_quality(img)
        if ok:
            st.markdown(f"<div class='step-content' style='color:#4CAF50;'>✅ {msg_en} | {msg_ur}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='step-content' style='color:#F44336;'>❌ {msg_en} | {msg_ur}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if not ok:
            st.stop()

        # ── STEP 2 ────────────────────────────────────────────────────────
        st.markdown("""
        <div class='step-card'>
            <div class='step-header'>🌿 Step 2: Leaf Detection</div>
            <div class='step-header-ur'>پتے کی تشخیص</div>
        """, unsafe_allow_html=True)

        leaf_r, green_r, brown_r = check_leaf(img)
        if leaf_r < 0.08:
            st.markdown("<div class='step-content' style='color:#F44336;'>❌ No leaf detected. Please upload a potato leaf image. | کوئی پتہ نہیں ملا۔ آلو کے پتے کی تصویر اپ لوڈ کریں۔</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        elif leaf_r < 0.15:
            st.markdown(f"<div class='step-content' style='color:#FF9800;'>⚠️ Leaf detected but not very clear ({leaf_r*100:.1f}%) | پتہ ملا لیکن واضح نہیں</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='step-content' style='color:#4CAF50;'>✅ Potato leaf detected ({leaf_r*100:.1f}% coverage) | آلو کا پتہ مل گیا</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── STEP 3 ────────────────────────────────────────────────────────
        st.markdown("""
        <div class='step-card'>
            <div class='step-header'>🧠 Step 3: Disease Detection</div>
            <div class='step-header-ur'>بیماری کی تشخیص</div>
        """, unsafe_allow_html=True)

        with st.spinner("Running CNN model..."):
            img_r = img.resize((256, 256))
            arr = np.expand_dims(np.array(img_r), axis=0)
            pred = model.predict(arr, verbose=0)

        conf = float(np.max(pred)) * 100
        result = classes[np.argmax(pred)]
        st.markdown(f"<div class='step-content' style='color:#C8A96E;'>CNN model processed successfully | ماڈل نے تصویر پروسیس کر لی</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── STEP 4 ────────────────────────────────────────────────────────
        st.markdown("""
        <div class='step-card'>
            <div class='step-header'>📊 Step 4: Confidence Score | اعتماد کا اسکور</div>
        """, unsafe_allow_html=True)

        bar_color = "#4CAF50" if conf >= 90 else "#FF9800" if conf >= 80 else "#F44336"
        st.markdown(f"""
        <div class='conf-bar-bg'>
            <div class='conf-bar-fill' style='width:{conf:.0f}%; background:{bar_color};'>
                {conf:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if conf < 80:
            st.markdown("""
            <div class='result-uncertain'>
                <p class='result-title-en'>⚠️ Unable to Confidently Identify</p>
                <p class='result-title-ur'>پہچان میں یقین نہیں</p>
                <p class='result-confidence'>Confidence below 80% threshold. Please upload a clearer image.<br>
                اعتماد 80% سے کم ہے۔ براہ کرم واضح تصویر اپ لوڈ کریں۔</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ── STEP 5 ────────────────────────────────────────────────────────
        sev_text, sev_color = get_severity(conf, result, brown_r)
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-header'>🔴 Step 5: Severity | شدت</div>
            <div style='margin-top:8px;'>
                <span class='severity-badge' style='background:{sev_color}22; border:1px solid {sev_color}; color:{sev_color};'>
                    {sev_text}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── RESULT ────────────────────────────────────────────────────────
        if result == "Healthy":
            st.markdown(f"""
            <div class='result-healthy'>
                <p class='result-title-en'>✅ Healthy Leaf</p>
                <p class='result-title-ur'>پتہ بالکل صحت مند ہے</p>
                <p class='result-confidence'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>💊 Step 6: Treatment | علاج</div>
                <div class='step-header-ur'>کوئی علاج ضروری نہیں</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>🛡️ Step 7: Prevention Tips | احتیاطی تدابیر</div>
                <div class='step-header-ur'></div>
            """, unsafe_allow_html=True)
            tips = [
                ("Monitor leaves every 3–5 days", "ہر 3-5 دن میں پتوں کا معائنہ کریں"),
                ("Water at base, not on leaves", "پتوں پر نہیں، جڑ میں پانی دیں"),
                ("Ensure good air circulation", "ہوا کی آمدورفت یقینی بنائیں"),
                ("Rotate crops every season", "ہر موسم میں فصل بدلیں"),
                ("Remove dead leaves from field", "خشک پتے کھیت سے ہٹائیں"),
            ]
            for en, ur in tips:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🌱 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif result == "Early Blight":
            st.markdown(f"""
            <div class='result-early'>
                <p class='result-title-en'>⚠️ Early Blight Detected</p>
                <p class='result-title-ur'>ابتدائی جھلساؤ کی بیماری پائی گئی</p>
                <p class='result-confidence'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>💊 Step 6: Treatment | علاج</div>
                <div class='step-header-ur'>جلد اقدام کریں</div>
            """, unsafe_allow_html=True)
            treatments = [
                ("Spray Mancozeb or Chlorothalonil fungicide immediately", "فوری طور پر مینکوزیب یا کلوروتھالونل فنگیسائیڈ چھڑکیں"),
                ("Remove infected leaves carefully", "متاثرہ پتے احتیاط سے ہٹائیں"),
                ("Do not compost infected leaves — burn them", "متاثرہ پتوں کو جلائیں، کھاد نہ بنائیں"),
                ("Avoid overhead watering", "اوپر سے پانی دینے سے پرہیز کریں"),
                ("Repeat treatment every 7–10 days", "علاج ہر 7-10 دن بعد دہرائیں"),
            ]
            for en, ur in treatments:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🧪 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>🛡️ Step 7: Prevention | احتیاط</div>
            """, unsafe_allow_html=True)
            prevs = [
                ("Use disease-resistant potato varieties", "بیماری سے بچنے والی اقسام استعمال کریں"),
                ("Practice crop rotation every 2–3 years", "ہر 2-3 سال میں فصل بدلیں"),
                ("Maintain proper plant spacing", "پودوں کے درمیان مناسب فاصلہ رکھیں"),
                ("Apply preventive fungicide before rainy season", "بارش سے پہلے فنگیسائیڈ لگائیں"),
            ]
            for en, ur in prevs:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:  # Late Blight
            st.markdown(f"""
            <div class='result-late'>
                <p class='result-title-en'>🚨 Late Blight Detected</p>
                <p class='result-title-ur'>دیر سے آنے والی جھلساؤ بیماری پائی گئی</p>
                <p class='result-confidence'>Confidence: {conf:.2f}% — Act IMMEDIATELY!</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>💊 Step 6: Treatment | علاج — فوری اقدام ضروری!</div>
            """, unsafe_allow_html=True)
            treatments = [
                ("Consult an agricultural expert immediately", "فوری طور پر زرعی ماہر سے رابطہ کریں"),
                ("Apply Metalaxyl or Cymoxanil fungicide right away", "میٹالیکسل یا سیموکسانل فوری لگائیں"),
                ("Remove and destroy ALL infected plants", "تمام متاثرہ پودے فوری ہٹائیں اور تلف کریں"),
                ("Isolate affected area immediately", "متاثرہ حصہ فوری الگ کریں"),
                ("Stop all overhead irrigation", "اوپر سے پانی دینا فوری بند کریں"),
                ("Repeat fungicide every 5–7 days", "ہر 5-7 دن بعد فنگیسائیڈ دہرائیں"),
            ]
            for en, ur in treatments:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🚑 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='step-card'>
                <div class='step-header'>🛡️ Step 7: Prevention | احتیاط</div>
            """, unsafe_allow_html=True)
            prevs = [
                ("Plant certified Late Blight-resistant varieties", "تصدیق شدہ بیماری مزاحم اقسام لگائیں"),
                ("Monitor weather — spreads fast in cool, wet conditions", "موسم کی نگرانی کریں — ٹھنڈے نم موسم میں تیزی سے پھیلتا ہے"),
                ("Never plant potatoes in the same field twice", "ایک کھیت میں دوبارہ آلو نہ لگائیں"),
                ("Remove all crop debris after harvest", "فصل کے بعد تمام باقیات ہٹائیں"),
            ]
            for en, ur in prevs:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class='step-card' style='text-align:center; padding:50px 20px;'>
            <p style='font-size:48px; margin:0;'>🥔</p>
            <p style='color:#F5DEB3; font-size:18px; margin:12px 0 4px 0;'>Upload a potato leaf image to begin analysis</p>
            <p style='color:#A8D878; font-size:16px; direction:rtl; margin:0;'>تجزیہ شروع کرنے کے لیے آلو کے پتے کی تصویر اپ لوڈ کریں</p>
            <br>
            <p style='color:#8B7355; font-size:13px;'>Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='earthy-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#5A4020; font-size:12px;'>
PotatoCare AI &nbsp;|&nbsp; Developed for Pak Angels Hackathon
</p>
""", unsafe_allow_html=True)
