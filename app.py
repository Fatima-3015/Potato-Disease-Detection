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
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #FFFFFF;
}

.stApp {
    background: #F8FFF8;
}

/* ── Hero ── */
.hero-box {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
    border-radius: 16px;
    padding: 36px 40px;
    text-align: center;
    margin-bottom: 24px;
}
.hero-title {
    font-family: 'Merriweather', serif;
    font-size: 44px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 8px 0;
}
.hero-sub-en {
    font-size: 16px;
    color: #C8E6C9;
    margin: 4px 0;
}
.hero-sub-ur {
    font-size: 17px;
    color: #A5D6A7;
    direction: rtl;
    margin: 4px 0;
}
.hero-tag {
    font-size: 12px;
    color: #81C784;
    margin-top: 10px;
}

/* ── Pipeline ── */
.pipeline-box {
    background: #FFFFFF;
    border: 2px solid #C8E6C9;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 24px;
}
.pipeline-title {
    font-size: 15px;
    font-weight: 600;
    color: #1B5E20;
    margin-bottom: 12px;
}
.pipeline-title-ur {
    font-size: 14px;
    color: #388E3C;
    direction: rtl;
    margin-bottom: 10px;
}
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.pipe-step {
    background: #E8F5E9;
    border: 1px solid #A5D6A7;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #1B5E20;
    font-weight: 600;
}
.pipe-arrow {
    color: #4CAF50;
    font-size: 16px;
    font-weight: bold;
}

/* ── Upload Card ── */
.upload-card {
    background: #FFFFFF;
    border: 2px dashed #4CAF50;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}
.upload-title {
    color: #1B5E20;
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 2px;
}
.upload-title-ur {
    color: #388E3C;
    font-size: 15px;
    direction: rtl;
}

/* ── Step Cards ── */
.step-card {
    background: #FFFFFF;
    border: 1px solid #C8E6C9;
    border-left: 4px solid #4CAF50;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
}
.step-header {
    font-size: 14px;
    font-weight: 700;
    color: #1B5E20;
    margin-bottom: 2px;
}
.step-header-ur {
    font-size: 13px;
    color: #388E3C;
    direction: rtl;
    margin-bottom: 6px;
}
.step-content-ok {
    font-size: 13px;
    color: #2E7D32;
}
.step-content-warn {
    font-size: 13px;
    color: #E65100;
}
.step-content-err {
    font-size: 13px;
    color: #B71C1C;
}

/* ── Result Boxes ── */
.result-healthy {
    background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
    border: 2px solid #4CAF50;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 10px 0;
}
.result-early {
    background: linear-gradient(135deg, #FFF8E1, #FFF3E0);
    border: 2px solid #FF9800;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 10px 0;
}
.result-late {
    background: linear-gradient(135deg, #FFEBEE, #FCE4EC);
    border: 2px solid #F44336;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 10px 0;
}
.result-uncertain {
    background: linear-gradient(135deg, #FFFDE7, #FFF9C4);
    border: 2px solid #FFC107;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 10px 0;
}
.result-title-en {
    font-size: 26px;
    font-weight: 700;
    color: #1A1A1A;
    margin: 0;
}
.result-title-ur {
    font-size: 18px;
    direction: rtl;
    margin: 4px 0;
    color: #333333;
}
.result-conf {
    font-size: 13px;
    color: #555555;
    margin-top: 6px;
}

/* ── Confidence Bar ── */
.conf-bar-bg {
    background: #E8F5E9;
    border-radius: 20px;
    height: 22px;
    margin: 8px 0;
    overflow: hidden;
    border: 1px solid #C8E6C9;
}
.conf-bar-fill {
    height: 22px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: white;
}

/* ── Severity Badge ── */
.sev-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 6px 4px;
}

/* ── Advice Cards ── */
.advice-card {
    background: #F9FBE7;
    border-left: 3px solid #8BC34A;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
}
.advice-en {
    font-size: 13px;
    color: #1B5E20;
    margin-bottom: 2px;
    font-weight: 500;
}
.advice-ur {
    font-size: 13px;
    color: #33691E;
    direction: rtl;
}

/* ── Divider ── */
.green-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4CAF50, transparent);
    margin: 18px 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2E7D32, #388E3C) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
}

/* ── File uploader ── */
section[data-testid="stFileUploadDropzone"] {
    background: #F1F8E9 !important;
    border: 2px dashed #4CAF50 !important;
    border-radius: 10px !important;
}

/* ── Image ── */
div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 2px solid #C8E6C9;
}

/* ── Placeholder ── */
.placeholder-box {
    background: #FFFFFF;
    border: 2px solid #C8E6C9;
    border-radius: 14px;
    padding: 60px 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-box'>
    <p class='hero-title'>🥔 PotatoCare AI</p>
    <p class='hero-sub-en'>Real-World Potato Disease & Crop Health Assistant</p>
    <p class='hero-sub-ur'>آلو کی بیماریوں کی تشخیص اور فصل کی صحت کا نظام</p>
    <p class='hero-tag'>Developed for Pak Angels Hackathon &nbsp;|&nbsp; Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='pipeline-box'>
    <div class='pipeline-title'>🔬 Detection Pipeline &nbsp;|&nbsp; <span style='color:#388E3C;'>تشخیص کا طریقہ کار</span></div>
    <div class='pipeline-row'>
        <span class='pipe-step'>📷 Image Quality</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🌿 Leaf Check</span>
        <span class='pipe-arrow'>→</span>
        <span class='pipe-step'>🧠 Disease Detection</span>
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

st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)

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
        return False, "Image too dark — use better lighting", "تصویر بہت تاریک ہے"
    if brightness > 245:
        return False, "Image overexposed — avoid direct sunlight", "تصویر بہت روشن ہے"
    return True, "Image quality is good", "تصویر کا معیار ٹھیک ہے"

def check_leaf(img):
    arr = np.array(img.convert('RGB'))
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green = np.sum((g > r) & (g > b) & (g > 40))
    brown = np.sum((r > 80) & (g > 50) & (b < 80) & (r > g))
    total = arr.shape[0] * arr.shape[1]
    return (green + brown) / total, green / total, brown / total

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

        # STEP 1
        st.markdown("<div class='step-card'><div class='step-header'>📷 Step 1: Image Quality Check</div><div class='step-header-ur'>تصویر کے معیار کی جانچ</div>", unsafe_allow_html=True)
        ok, msg_en, msg_ur = check_quality(img)
        if ok:
            st.markdown(f"<div class='step-content-ok'>✅ {msg_en} | {msg_ur}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='step-content-err'>❌ {msg_en} | {msg_ur}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if not ok:
            st.stop()

        # STEP 2
        st.markdown("<div class='step-card'><div class='step-header'>🌿 Step 2: Leaf Detection</div><div class='step-header-ur'>پتے کی تشخیص</div>", unsafe_allow_html=True)
        leaf_r, green_r, brown_r = check_leaf(img)
        if leaf_r < 0.08:
            st.markdown("<div class='step-content-err'>❌ No leaf detected. Please upload a potato leaf image. | کوئی پتہ نہیں ملا</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        elif leaf_r < 0.15:
            st.markdown(f"<div class='step-content-warn'>⚠️ Leaf detected but not very clear ({leaf_r*100:.1f}%) | پتہ ملا لیکن واضح نہیں</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='step-content-ok'>✅ Potato leaf detected ({leaf_r*100:.1f}% coverage) | آلو کا پتہ مل گیا</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 3
        st.markdown("<div class='step-card'><div class='step-header'>🧠 Step 3: Disease Detection</div><div class='step-header-ur'>بیماری کی تشخیص</div>", unsafe_allow_html=True)
        with st.spinner("Running CNN model..."):
            img_r = img.resize((256, 256))
            arr = np.expand_dims(np.array(img_r), axis=0)
            pred = model.predict(arr, verbose=0)
        conf = float(np.max(pred)) * 100
        result = classes[np.argmax(pred)]
        st.markdown(f"<div class='step-content-ok'>✅ CNN model processed successfully | ماڈل نے تصویر پروسیس کر لی</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 4
        bar_color = "#4CAF50" if conf >= 90 else "#FF9800" if conf >= 80 else "#F44336"
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-header'>📊 Step 4: Confidence Score | اعتماد کا اسکور</div>
            <div class='conf-bar-bg'>
                <div class='conf-bar-fill' style='width:{conf:.0f}%; background:{bar_color};'>
                    {conf:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if conf < 80:
            st.markdown("""
            <div class='result-uncertain'>
                <p class='result-title-en'>⚠️ Unable to Confidently Identify</p>
                <p class='result-title-ur'>پہچان میں یقین نہیں</p>
                <p class='result-conf'>Confidence below 80%. Please upload a clearer image.<br>
                اعتماد 80% سے کم ہے۔ براہ کرم واضح تصویر اپ لوڈ کریں۔</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # STEP 5
        sev_text, sev_color = get_severity(conf, result, brown_r)
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-header'>🔴 Step 5: Severity | شدت</div>
            <span class='sev-badge' style='background:{sev_color}22; border:1px solid {sev_color}; color:{sev_color};'>
                {sev_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # RESULT
        if result == "Healthy":
            st.markdown(f"""
            <div class='result-healthy'>
                <p class='result-title-en'>✅ Healthy Leaf</p>
                <p class='result-title-ur'>پتہ بالکل صحت مند ہے</p>
                <p class='result-conf'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='step-card'><div class='step-header'>💊 Step 6: Treatment | علاج</div>", unsafe_allow_html=True)
            st.markdown("<div class='advice-card'><div class='advice-en'>✅ No treatment needed — your plant is perfectly healthy!</div><div class='advice-ur'>کوئی علاج ضروری نہیں — آپ کا پودا بالکل صحت مند ہے</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='step-card'><div class='step-header'>🛡️ Step 7: Prevention | احتیاطی تدابیر</div>", unsafe_allow_html=True)
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
                <p class='result-conf'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='step-card'><div class='step-header'>💊 Step 6: Treatment | علاج</div>", unsafe_allow_html=True)
            treatments = [
                ("Spray Mancozeb or Chlorothalonil fungicide immediately", "فوری طور پر مینکوزیب فنگیسائیڈ چھڑکیں"),
                ("Remove infected leaves carefully", "متاثرہ پتے احتیاط سے ہٹائیں"),
                ("Do not compost infected leaves — burn them", "متاثرہ پتوں کو جلائیں، کھاد نہ بنائیں"),
                ("Avoid overhead watering", "اوپر سے پانی دینے سے پرہیز کریں"),
                ("Repeat treatment every 7–10 days", "علاج ہر 7-10 دن بعد دہرائیں"),
            ]
            for en, ur in treatments:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🧪 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='step-card'><div class='step-header'>🛡️ Step 7: Prevention | احتیاط</div>", unsafe_allow_html=True)
            prevs = [
                ("Use disease-resistant potato varieties", "بیماری سے بچنے والی اقسام استعمال کریں"),
                ("Practice crop rotation every 2–3 years", "ہر 2-3 سال میں فصل بدلیں"),
                ("Maintain proper plant spacing", "پودوں کے درمیان مناسب فاصلہ رکھیں"),
                ("Apply preventive fungicide before rainy season", "بارش سے پہلے فنگیسائیڈ لگائیں"),
            ]
            for en, ur in prevs:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='result-late'>
                <p class='result-title-en'>🚨 Late Blight Detected</p>
                <p class='result-title-ur'>دیر سے آنے والی جھلساؤ بیماری پائی گئی</p>
                <p class='result-conf'>Confidence: {conf:.2f}% — Act IMMEDIATELY!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='step-card'><div class='step-header'>💊 Step 6: Treatment | علاج — فوری اقدام!</div>", unsafe_allow_html=True)
            treatments = [
                ("Consult an agricultural expert immediately", "فوری طور پر زرعی ماہر سے رابطہ کریں"),
                ("Apply Metalaxyl or Cymoxanil fungicide right away", "میٹالیکسل یا سیموکسانل فوری لگائیں"),
                ("Remove and destroy ALL infected plants", "تمام متاثرہ پودے فوری ہٹائیں"),
                ("Isolate affected area immediately", "متاثرہ حصہ فوری الگ کریں"),
                ("Stop all overhead irrigation", "اوپر سے پانی دینا فوری بند کریں"),
                ("Repeat fungicide every 5–7 days", "ہر 5-7 دن بعد فنگیسائیڈ دہرائیں"),
            ]
            for en, ur in treatments:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🚑 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='step-card'><div class='step-header'>🛡️ Step 7: Prevention | احتیاط</div>", unsafe_allow_html=True)
            prevs = [
                ("Plant certified Late Blight-resistant varieties", "تصدیق شدہ بیماری مزاحم اقسام لگائیں"),
                ("Monitor weather — spreads fast in cool, wet conditions", "موسم کی نگرانی کریں"),
                ("Never plant potatoes in the same field twice", "ایک کھیت میں دوبارہ آلو نہ لگائیں"),
                ("Remove all crop debris after harvest", "فصل کے بعد تمام باقیات ہٹائیں"),
            ]
            for en, ur in prevs:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class='placeholder-box'>
            <p style='font-size:52px; margin:0;'>🥔</p>
            <p style='color:#1B5E20; font-size:17px; font-weight:600; margin:12px 0 4px 0;'>
                Upload a potato leaf image to begin
            </p>
            <p style='color:#388E3C; font-size:15px; direction:rtl; margin:0;'>
                تجزیہ شروع کرنے کے لیے تصویر اپ لوڈ کریں
            </p>
            <p style='color:#81C784; font-size:12px; margin-top:12px;'>Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#81C784; font-size:12px;'>
PotatoCare AI &nbsp;|&nbsp; Developed for Pak Angels Hackathon
</p>
""", unsafe_allow_html=True)