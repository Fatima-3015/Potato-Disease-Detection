import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import hashlib
import os

st.set_page_config(
    page_title="PotatoCare AI",
    page_icon="🥔",
    layout="wide"
)

# ── TEST / DUMMY PHONE NUMBER CONFIG ───────────────────────────────────────────
# This MUST match exactly what you register in Firebase Console under
# Authentication → Sign-in method → Phone → "Phone numbers for testing".
# See the setup instructions given alongside this file.
TEST_PHONE_NUMBER = "+923001234567"
TEST_PHONE_OTP = "123456"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
}
.stApp {
    background: #F8FFF8;
}

/* Hero */
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
.hero-sub-en { font-size: 16px; color: #C8E6C9; margin: 4px 0; }
.hero-sub-ur { font-size: 17px; color: #A5D6A7; direction: rtl; margin: 4px 0; }
.hero-tag { font-size: 12px; color: #81C784; margin-top: 10px; }

/* Pipeline */
.pipeline-box {
    background: #FFFFFF;
    border: 2px solid #C8E6C9;
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 24px;
}
.pipeline-title { font-size: 14px; font-weight: 700; color: #1B5E20; margin-bottom: 12px; }
.pipeline-row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.pipe-step {
    background: #E8F5E9;
    border: 1px solid #A5D6A7;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    color: #1B5E20;
    font-weight: 600;
}
.pipe-arrow { color: #4CAF50; font-size: 18px; font-weight: bold; }

/* Upload */
.upload-card {
    background: #FFFFFF;
    border: 2px dashed #4CAF50;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}
.upload-title { color: #1B5E20; font-size: 17px; font-weight: 600; margin-bottom: 2px; }
.upload-title-ur { color: #388E3C; font-size: 15px; direction: rtl; }

/* Result cards */
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
.result-title-en { font-size: 26px; font-weight: 700; color: #1A1A1A; margin: 0; }
.result-title-ur { font-size: 18px; direction: rtl; margin: 4px 0; color: #333; }
.result-conf { font-size: 13px; color: #555; margin-top: 6px; }

/* Check cards */
.check-card {
    background: #FFFFFF;
    border: 1px solid #C8E6C9;
    border-left: 4px solid #4CAF50;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}
.check-ok { color: #2E7D32; font-weight: 600; }
.check-warn { color: #E65100; font-weight: 600; }
.check-err { color: #B71C1C; font-weight: 600; }

/* Confidence bar */
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

/* Severity */
.sev-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 6px 0;
}

/* Advice */
.advice-card {
    background: #F9FBE7;
    border-left: 3px solid #8BC34A;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
}
.advice-en { font-size: 13px; color: #1B5E20; margin-bottom: 2px; font-weight: 500; }
.advice-ur { font-size: 13px; color: #33691E; direction: rtl; }

/* Section header */
.sec-header {
    font-size: 14px;
    font-weight: 700;
    color: #1B5E20;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 2px solid #C8E6C9;
}
.sec-header-ur {
    font-size: 13px;
    color: #388E3C;
    direction: rtl;
    margin-bottom: 8px;
}

/* Divider */
.green-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4CAF50, transparent);
    margin: 18px 0;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2E7D32, #388E3C) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
}

/* Hide default file uploader label */
.stFileUploader label { display: none !important; }

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 2px solid #C8E6C9;
}

.placeholder-box {
    background: #FFFFFF;
    border: 2px solid #C8E6C9;
    border-radius: 14px;
    padding: 60px 20px;
    text-align: center;
}

/* History card */
.history-card {
    background: #FFFFFF;
    border: 1px solid #C8E6C9;
    border-left: 4px solid #4CAF50;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 12px;
}
.history-date { color: #666; font-size: 11px; }
.history-result { font-weight: 700; font-size: 13px; }

/* Auth card */
.auth-banner {
    background: #FFFFFF;
    border: 1px solid #C8E6C9;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ── FIREBASE / FIRESTORE SETUP ─────────────────────────────────────────────────
@st.cache_resource
def init_firestore():
    if not firebase_admin._apps:
        cred_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore()
WEB_API_KEY = st.secrets["firebase_web_api_key"]
PROJECT_ID = st.secrets["firebase"]["project_id"]
AUTH_DOMAIN = f"{PROJECT_ID}.firebaseapp.com"

IDENTITY_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"

# ── PASSWORD HASHING (for phone accounts, stored in Firestore) ────────────────
def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 100000).hex()
    return pwd_hash, salt

def verify_password(password, salt, pwd_hash):
    check_hash, _ = hash_password(password, salt)
    return check_hash == pwd_hash

# ── EMAIL AUTH (real Firebase Authentication via REST API) ────────────────────
def email_signup(email, password):
    r = requests.post(f"{IDENTITY_BASE}:signUp?key={WEB_API_KEY}", json={
        "email": email, "password": password, "returnSecureToken": True
    })
    data = r.json()
    if "error" in data:
        return False, data["error"].get("message", "Signup failed")
    id_token = data["idToken"]
    requests.post(f"{IDENTITY_BASE}:sendOobCode?key={WEB_API_KEY}", json={
        "requestType": "VERIFY_EMAIL", "idToken": id_token
    })
    return True, "Account created! Verification email sent. | اکاؤنٹ بن گیا! تصدیقی ای میل بھیج دی گئی ہے۔"

def email_login(email, password):
    r = requests.post(f"{IDENTITY_BASE}:signInWithPassword?key={WEB_API_KEY}", json={
        "email": email, "password": password, "returnSecureToken": True
    })
    data = r.json()
    if "error" in data:
        return False, data["error"].get("message", "Login failed"), False
    id_token = data["idToken"]
    lookup = requests.post(f"{IDENTITY_BASE}:lookup?key={WEB_API_KEY}", json={"idToken": id_token}).json()
    verified = lookup.get("users", [{}])[0].get("emailVerified", False)
    return True, "Login successful", verified

def email_forgot_password(email):
    r = requests.post(f"{IDENTITY_BASE}:sendOobCode?key={WEB_API_KEY}", json={
        "requestType": "PASSWORD_RESET", "email": email
    })
    data = r.json()
    if "error" in data:
        return False, data["error"].get("message", "Could not send reset email")
    return True, "Password reset email sent. | پاس ورڈ ری سیٹ ای میل بھیج دی گئی ہے۔"

# ── PHONE AUTH (Firestore-based, phone verified once via Firebase test OTP) ──
def phone_user_exists(phone):
    docs = list(db.collection("phone_users").where("phone", "==", phone).limit(1).stream())
    return docs[0].to_dict() if docs else None

def create_phone_user(phone, password):
    pwd_hash, salt = hash_password(password)
    db.collection("phone_users").add({
        "phone": phone, "password_hash": pwd_hash, "salt": salt, "created_at": datetime.now()
    })

def phone_login(phone, password):
    user = phone_user_exists(phone)
    if not user:
        return False, "This number is not registered. | یہ نمبر رجسٹرڈ نہیں ہے۔"
    if verify_password(password, user["salt"], user["password_hash"]):
        return True, "Login successful"
    return False, "Incorrect password. | غلط پاس ورڈ۔"

# ── HISTORY (Firestore) ────────────────────────────────────────────────────────
def save_history(user_id, filename, result, confidence, severity):
    db.collection("history").add({
        "user_name": user_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "timestamp_sort": datetime.now(),
        "filename": filename,
        "result": result,
        "confidence": confidence,
        "severity": severity,
    })

def get_history(user_id, limit=15):
    docs = db.collection("history").where("user_name", "==", user_id).stream()
    records = [d.to_dict() for d in docs]
    records.sort(key=lambda r: r.get("timestamp_sort", datetime.min), reverse=True)
    return records[:limit]

# ── SESSION STATE DEFAULTS ─────────────────────────────────────────────────────
if "auth_status" not in st.session_state:
    st.session_state.auth_status = None   # None | "guest" | "authed"
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "phone_verified_pending" not in st.session_state:
    st.session_state.phone_verified_pending = None

# Pick up phone verification redirect (from the JS OTP widget)
qp = st.query_params
if "verified_phone" in qp:
    st.session_state.phone_verified_pending = qp["verified_phone"]
    st.query_params.clear()

# ── SIDEBAR: AUTH ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 👤 Account | اکاؤنٹ")

if st.session_state.auth_status == "authed":
    st.sidebar.markdown(f"Welcome back, **{st.session_state.user_id}**! 👋")
    if st.sidebar.button("Logout"):
        st.session_state.auth_status = None
        st.session_state.user_id = None
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Your Upload History")
    past_records = get_history(st.session_state.user_id)
    if past_records:
        result_colors = {"Healthy": "#4CAF50", "Early Blight": "#FF9800", "Late Blight": "#F44336"}
        for rec in past_records:
            color = result_colors.get(rec.get("result"), "#666")
            st.sidebar.markdown(f"""
            <div class='history-card'>
                <div class='history-date'>🕒 {rec.get('timestamp', '')}</div>
                <div class='history-result' style='color:{color};'>{rec.get('result', '')} — {rec.get('confidence', 0):.1f}%</div>
                <div style='font-size:11px; color:#888;'>{rec.get('filename', '')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<p style='font-size:13px; color:#888;'>No history found — upload your first image. | ابھی کوئی ہسٹری نہیں — پہلی تصویر اپ لوڈ کریں۔</p>", unsafe_allow_html=True)

elif st.session_state.auth_status == "guest":
    st.sidebar.markdown("<div class='auth-banner'>👤 Guest mode — history will not be saved. | مہمان موڈ — ہسٹری محفوظ نہیں ہوگی۔</div>", unsafe_allow_html=True)
    if st.sidebar.button("Sign up / Login instead"):
        st.session_state.auth_status = None
        st.rerun()

else:
    tab_login, tab_signup, tab_guest = st.sidebar.tabs(["Login", "Sign up", "Guest"])

    # ── LOGIN TAB ──
    with tab_login:
        login_method = st.radio("Login with", ["Email", "Phone"], key="login_method", horizontal=True)
        if login_method == "Email":
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pw")
            if st.button("Login", key="login_btn_email"):
                ok, msg, verified = email_login(email, password)
                if ok:
                    if verified:
                        st.session_state.auth_status = "authed"
                        st.session_state.user_id = email
                        st.rerun()
                    else:
                        st.warning("Email not verified yet. Please check your inbox and click the verification link. | ای میل تصدیق شدہ نہیں۔ براہ کرم اپنا ان باکس چیک کریں۔")
                else:
                    st.error(msg)
            with st.expander("Forgot password?"):
                fp_email = st.text_input("Enter your email | اپنا ای میل لکھیں", key="fp_email")
                if st.button("Send reset link", key="fp_btn"):
                    ok, msg = email_forgot_password(fp_email)
                    st.success(msg) if ok else st.error(msg)
        else:
            phone = st.text_input("Phone number (e.g. +923001234567)", key="login_phone")
            password = st.text_input("Password", type="password", key="login_pw_phone")
            if st.button("Login", key="login_btn_phone"):
                ok, msg = phone_login(phone, password)
                if ok:
                    st.session_state.auth_status = "authed"
                    st.session_state.user_id = phone
                    st.rerun()
                else:
                    st.error(msg)

    # ── SIGNUP TAB ──
    with tab_signup:
        signup_method = st.radio("Sign up with", ["Email", "Phone"], key="signup_method", horizontal=True)
        if signup_method == "Email":
            new_email = st.text_input("Email", key="signup_email")
            new_pw = st.text_input("Password", type="password", key="signup_pw")
            confirm_pw = st.text_input("Confirm password", type="password", key="signup_pw2")
            if st.button("Create account", key="signup_btn_email"):
                if new_pw != confirm_pw:
                    st.error("Passwords do not match. | پاس ورڈ میچ نہیں ہو رہے۔")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters. | پاس ورڈ کم از کم 6 حروف کا ہونا چاہیے۔")
                else:
                    ok, msg = email_signup(new_email, new_pw)
                    st.success(msg) if ok else st.error(msg)
        else:
            if not st.session_state.phone_verified_pending:
                # ── Dummy/Test number banner ──
                st.info(
                    f"🧪 **Demo mode:** use test number **{TEST_PHONE_NUMBER}** "
                    f"with OTP **{TEST_PHONE_OTP}** — no real SMS will be sent.\n\n"
                    f"یہ ٹیسٹ نمبر ہے، کوئی اصل SMS نہیں بھیجا جائے گا۔"
                )
                st.components.v1.html(f"""
                <div id="recaptcha-container"></div>
                <input id="phone-input" placeholder="+923001234567" value="{TEST_PHONE_NUMBER}" style="width:100%;padding:8px;margin-bottom:6px;border-radius:6px;border:1px solid #ccc;">
                <button id="send-otp-btn" style="width:100%;padding:8px;background:#2E7D32;color:white;border:none;border-radius:6px;margin-bottom:6px;">Send OTP</button>
                <div id="otp-section" style="display:none;">
                  <input id="otp-input" placeholder="Enter OTP (e.g. {TEST_PHONE_OTP})" style="width:100%;padding:8px;margin-bottom:6px;border-radius:6px;border:1px solid #ccc;">
                  <button id="verify-otp-btn" style="width:100%;padding:8px;background:#1B5E20;color:white;border:none;border-radius:6px;">Verify OTP</button>
                </div>
                <div id="status-msg" style="font-size:12px;margin-top:6px;"></div>
                <script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-app-compat.js"></script>
                <script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-auth-compat.js"></script>
                <script>
                  const firebaseConfig = {{
                    apiKey: "{WEB_API_KEY}",
                    authDomain: "{AUTH_DOMAIN}",
                    projectId: "{PROJECT_ID}"
                  }};
                  firebase.initializeApp(firebaseConfig);
                  window.recaptchaVerifier = new firebase.auth.RecaptchaVerifier('recaptcha-container', {{ size: 'invisible' }});
                  let confirmationResult;
                  document.getElementById('send-otp-btn').onclick = function() {{
                    const phone = document.getElementById('phone-input').value;
                    firebase.auth().signInWithPhoneNumber(phone, window.recaptchaVerifier)
                      .then((result) => {{
                        confirmationResult = result;
                        document.getElementById('otp-section').style.display = 'block';
                        document.getElementById('status-msg').innerText = 'OTP sent (for the test number, use the fixed code: {TEST_PHONE_OTP})';
                      }}).catch((error) => {{
                        document.getElementById('status-msg').innerText = 'Error: ' + error.message;
                      }});
                  }};
                  document.getElementById('verify-otp-btn').onclick = function() {{
                    const code = document.getElementById('otp-input').value;
                    confirmationResult.confirm(code).then((result) => {{
                      const phone = result.user.phoneNumber;
                      window.top.location.href = window.top.location.pathname + '?verified_phone=' + encodeURIComponent(phone);
                    }}).catch((error) => {{
                      document.getElementById('status-msg').innerText = 'Incorrect OTP, please try again';
                    }});
                  }};
                </script>
                """, height=260)
            else:
                verified_phone = st.session_state.phone_verified_pending
                st.success(f"✅ {verified_phone} verified! Now set a password. | تصدیق ہو گئی! اب پاس ورڈ سیٹ کریں۔")
                new_pw = st.text_input("Set Password | پاس ورڈ سیٹ کریں", type="password", key="phone_signup_pw")
                confirm_pw = st.text_input("Confirm password", type="password", key="phone_signup_pw2")
                if st.button("Create Account | اکاؤنٹ بنائیں", key="phone_signup_btn"):
                    if new_pw != confirm_pw:
                        st.error("Passwords do not match. | پاس ورڈ میچ نہیں ہو رہے۔")
                    elif len(new_pw) < 6:
                        st.error("Password must be at least 6 characters. | پاس ورڈ کم از کم 6 حروف کا ہونا چاہیے۔")
                    elif phone_user_exists(verified_phone):
                        st.error("This number is already registered. | یہ نمبر پہلے سے رجسٹرڈ ہے۔")
                    else:
                        create_phone_user(verified_phone, new_pw)
                        st.session_state.phone_verified_pending = None
                        st.success("Account created! Please sign in from the Login tab. | اکاؤنٹ بن گیا! اب لاگ ان ٹیب سے سائن ان کریں۔")

    # ── GUEST TAB ──
    with tab_guest:
        st.markdown("<p style='font-size:13px; color:#666;'>Use the app without signing up — you'll get results but history won't be saved. | بغیر سائن اپ کے ایپ استعمال کریں — نتیجہ ملے گا لیکن ہسٹری محفوظ نہیں ہوگی۔</p>", unsafe_allow_html=True)
        if st.button("Continue as Guest", key="guest_btn"):
            st.session_state.auth_status = "guest"
            st.rerun()

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
        <span class='pipe-arrow'>➜</span>
        <span class='pipe-step'>🌿 Leaf Check</span>
        <span class='pipe-arrow'>➜</span>
        <span class='pipe-step'>🧠 Disease Detection</span>
        <span class='pipe-arrow'>➜</span>
        <span class='pipe-step'>📊 Confidence</span>
        <span class='pipe-arrow'>➜</span>
        <span class='pipe-step'>🔴 Severity</span>
        <span class='pipe-arrow'>➜</span>
        <span class='pipe-step'>💊 Treatment</span>
        <span class='pipe-arrow'>➜</span>
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

# ── LAYOUT (single column: upload/instructions on top, results below) ─────────
st.markdown("""
<div class='upload-card'>
    <p class='upload-title'>📁 Upload Potato Leaf Image</p>
    <p class='upload-title-ur'>آلو کے پتے کی تصویر اپ لوڈ کریں</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

can_analyze = st.session_state.auth_status in ("authed", "guest")

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if not can_analyze:
        st.markdown("<div class='check-card'><span class='check-warn'>⚠️ Please choose Login/Signup/Guest from the sidebar first | براہ کرم پہلے سائیڈبار سے آپشن منتخب کریں</span></div>", unsafe_allow_html=True)
    analyze = st.button("🔍 Analyze Now | ابھی تجزیہ کریں", disabled=not can_analyze)
else:
    analyze = False

st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)

if True:
    if not uploaded_file:
        st.markdown("""
        <div class='placeholder-box'>
            <p style='font-size:52px; margin:0;'>🥔</p>
            <p style='color:#1B5E20; font-size:17px; font-weight:600; margin:12px 0 4px 0;'>
                1st upload the picture then analyze
            </p>
            <p style='color:#388E3C; font-size:15px; direction:rtl; margin:0;'>
                پہلے تصویر اپ لوڈ کریں، پھر تجزیہ کریں
            </p>
            <p style='color:#81C784; font-size:12px; margin-top:12px;'>Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and not analyze:
        st.markdown("""
        <div class='placeholder-box'>
            <p style='font-size:40px; margin:0;'>👆</p>
            <p style='color:#1B5E20; font-size:16px; font-weight:600; margin:12px 0 4px 0;'>
                1st upload the picture then analyze
            </p>
            <p style='color:#388E3C; font-size:14px; direction:rtl; margin:0;'>
                تصویر اپ لوڈ ہو گئی، اب "Analyze Now" دبائیں
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and analyze:

        # ── Quality Check ─────────────────────────────────────────────────
        ok, msg_en, msg_ur = check_quality(img)
        cls = "check-ok" if ok else "check-err"
        icon = "✅" if ok else "❌"
        st.markdown(f"<div class='check-card'><span class='{cls}'>{icon} {msg_en}</span> &nbsp;|&nbsp; <span style='direction:rtl; color:#388E3C;'>{msg_ur}</span></div>", unsafe_allow_html=True)
        if not ok:
            st.stop()

        # ── Leaf Check ────────────────────────────────────────────────────
        leaf_r, green_r, brown_r = check_leaf(img)
        if leaf_r < 0.08:
            st.markdown("<div class='check-card'><span class='check-err'>❌ No leaf detected. Please upload a potato leaf image. | کوئی پتہ نہیں ملا</span></div>", unsafe_allow_html=True)
            st.stop()
        elif leaf_r < 0.15:
            st.markdown(f"<div class='check-card'><span class='check-warn'>⚠️ Leaf detected but not very clear ({leaf_r*100:.1f}%) | پتہ ملا لیکن واضح نہیں</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='check-card'><span class='check-ok'>✅ Potato leaf detected ({leaf_r*100:.1f}%) | آلو کا پتہ مل گیا</span></div>", unsafe_allow_html=True)

        # ── CNN (hidden) ──────────────────────────────────────────────────
        with st.spinner("Analyzing..."):
            img_r = img.resize((256, 256))
            arr = np.expand_dims(np.array(img_r), axis=0)
            pred = model.predict(arr, verbose=0)
        conf = float(np.max(pred)) * 100
        result = classes[np.argmax(pred)]

        # ── Confidence ────────────────────────────────────────────────────
        bar_color = "#4CAF50" if conf >= 90 else "#FF9800" if conf >= 80 else "#F44336"
        st.markdown(f"""
        <div class='check-card'>
            <span style='font-size:13px; font-weight:700; color:#1B5E20;'>📊 Confidence | اعتماد</span>
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
                <p class='result-conf'>Please upload a clearer potato leaf image.<br>براہ کرم واضح تصویر اپ لوڈ کریں</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ── Severity ──────────────────────────────────────────────────────
        sev_text, sev_color = get_severity(conf, result, brown_r)
        st.markdown(f"""
        <div class='check-card'>
            <span style='font-size:13px; font-weight:700; color:#1B5E20;'>🔴 Severity | شدت &nbsp;</span>
            <span class='sev-badge' style='background:{sev_color}22; border:1px solid {sev_color}; color:{sev_color};'>
                {sev_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Save to history (only for authed users, not guests) ────────────
        if st.session_state.auth_status == "authed":
            save_history(st.session_state.user_id, uploaded_file.name, result, conf, sev_text)

        # ── Result ────────────────────────────────────────────────────────
        if result == "Healthy":
            st.markdown(f"""
            <div class='result-healthy'>
                <p class='result-title-en'>✅ Healthy Leaf</p>
                <p class='result-title-ur'>پتہ بالکل صحت مند ہے</p>
                <p class='result-conf'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>💊 Treatment | علاج</div>", unsafe_allow_html=True)
            st.markdown("<div class='advice-card'><div class='advice-en'>✅ No treatment needed — your plant is perfectly healthy!</div><div class='advice-ur'>کوئی علاج ضروری نہیں — آپ کا پودا بالکل صحت مند ہے</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>🛡️ Prevention | احتیاطی تدابیر</div>", unsafe_allow_html=True)
            for en, ur in [
                ("Monitor leaves every 3–5 days", "ہر 3-5 دن میں پتوں کا معائنہ کریں"),
                ("Water at base, not on leaves", "پتوں پر نہیں، جڑ میں پانی دیں"),
                ("Ensure good air circulation", "ہوا کی آمدورفت یقینی بنائیں"),
                ("Rotate crops every season", "ہر موسم میں فصل بدلیں"),
                ("Remove dead leaves from field", "خشک پتے کھیت سے ہٹائیں"),
            ]:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🌱 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)

        elif result == "Early Blight":
            st.markdown(f"""
            <div class='result-early'>
                <p class='result-title-en'>⚠️ Early Blight Detected</p>
                <p class='result-title-ur'>ابتدائی جھلساؤ کی بیماری پائی گئی</p>
                <p class='result-conf'>Confidence: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>💊 Treatment | علاج</div>", unsafe_allow_html=True)
            for en, ur in [
                ("Spray Mancozeb or Chlorothalonil fungicide immediately", "فوری طور پر مینکوزیب فنگیسائیڈ چھڑکیں"),
                ("Remove infected leaves carefully", "متاثرہ پتے احتیاط سے ہٹائیں"),
                ("Do not compost infected leaves — burn them", "متاثرہ پتوں کو جلائیں، کھاد نہ بنائیں"),
                ("Avoid overhead watering", "اوپر سے پانی دینے سے پرہیز کریں"),
                ("Repeat treatment every 7–10 days", "علاج ہر 7-10 دن بعد دہرائیں"),
            ]:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🧪 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>🛡️ Prevention | احتیاط</div>", unsafe_allow_html=True)
            for en, ur in [
                ("Use disease-resistant potato varieties", "بیماری سے بچنے والی اقسام استعمال کریں"),
                ("Practice crop rotation every 2–3 years", "ہر 2-3 سال میں فصل بدلیں"),
                ("Maintain proper plant spacing", "پودوں کے درمیان مناسب فاصلہ رکھیں"),
                ("Apply preventive fungicide before rainy season", "بارش سے پہلے فنگیسائیڈ لگائیں"),
            ]:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='result-late'>
                <p class='result-title-en'>🚨 Late Blight Detected</p>
                <p class='result-title-ur'>دیر سے آنے والی جھلساؤ بیماری پائی گئی</p>
                <p class='result-conf'>Confidence: {conf:.2f}% — Act IMMEDIATELY!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>💊 Treatment | فوری علاج</div>", unsafe_allow_html=True)
            for en, ur in [
                ("Consult an agricultural expert immediately", "فوری طور پر زرعی ماہر سے رابطہ کریں"),
                ("Apply Metalaxyl or Cymoxanil fungicide right away", "میٹالیکسل یا سیموکسانل فوری لگائیں"),
                ("Remove and destroy ALL infected plants", "تمام متاثرہ پودے فوری ہٹائیں"),
                ("Isolate affected area immediately", "متاثرہ حصہ فوری الگ کریں"),
                ("Stop all overhead irrigation", "اوپر سے پانی دینا فوری بند کریں"),
                ("Repeat fungicide every 5–7 days", "ہر 5-7 دن بعد فنگیسائیڈ دہرائیں"),
            ]:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🚑 {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-header'>🛡️ Prevention | احتیاط</div>", unsafe_allow_html=True)
            for en, ur in [
                ("Plant certified Late Blight-resistant varieties", "تصدیق شدہ بیماری مزاحم اقسام لگائیں"),
                ("Monitor weather — spreads fast in cool wet conditions", "موسم کی نگرانی کریں"),
                ("Never plant potatoes in the same field twice", "ایک کھیت میں دوبارہ آلو نہ لگائیں"),
                ("Remove all crop debris after harvest", "فصل کے بعد تمام باقیات ہٹائیں"),
            ]:
                st.markdown(f"<div class='advice-card'><div class='advice-en'>🛡️ {en}</div><div class='advice-ur'>{ur}</div></div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#81C784; font-size:12px;'>
PotatoCare AI &nbsp;|&nbsp; Developed for Pak Angels Hackathon
</p>
""", unsafe_allow_html=True)