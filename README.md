# 🥔 PotatoCare AI
### Real-World Potato Disease & Crop Health Assistant

[![Live App](https://img.shields.io/badge/Live%20App-potatocare--ai.streamlit.app-brightgreen?style=for-the-badge&logo=streamlit)](https://potatocare-ai.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Fatima--3015-181717?style=for-the-badge&logo=github)](https://github.com/Fatima-3015/Potato-Disease-Detection)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Pak Angels](https://img.shields.io/badge/Pak%20Angels-Cohort%2011-gold?style=for-the-badge)](https://pakangels.com)

> **Developed for Pak Angels Generative & Agentic AI Hackathon — Cohort 11**  
> Delivered by: PAKANGELS | iCode Guru | ASPIRE Pakistan

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Detection Pipeline](#-detection-pipeline)
- [Technology Stack](#-technology-stack)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running Locally](#-running-locally)
- [Deployment](#-deployment)
- [Authentication](#-authentication)
- [Screenshots](#-screenshots)
- [Team](#-team)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🌿 About the Project

**PotatoCare AI** is a deployed, AI-powered web application that helps farmers and agricultural workers detect potato leaf diseases in real time. A user uploads a photograph of a potato leaf, and the system:

1. Validates image quality
2. Verifies a leaf is present
3. Runs a trained **Convolutional Neural Network (CNN)**
4. Returns one of three classifications: **Early Blight**, **Late Blight**, or **Healthy**
5. Shows a confidence score, severity assessment, and bilingual **English + Urdu** treatment and prevention advice

The app is live on **Streamlit Cloud** and includes **Firebase Authentication** (email + phone OTP), upload history tracking via **Firebase Firestore**, and a seven-step analysis pipeline.

---

## ❗ Problem Statement

Potato crops are among the most economically important food crops in Pakistan and globally. They are highly susceptible to two devastating fungal diseases:

- **Early Blight** (*Alternaria solani*) — dark circular spots with yellow halo on older leaves
- **Late Blight** (*Phytophthora infestans*) — water-soaked lesions that can destroy an entire field within days

**Core problems this project solves:**

| Problem | Impact |
|---|---|
| Farmers in rural areas have no access to agricultural experts | Late diagnosis, massive crop losses |
| Manual visual inspection is slow and subjective | Disease spreads before treatment begins |
| Existing tools are English-only | Language barrier for Urdu-speaking farmers |
| No free, accessible, smartphone-compatible tool exists | Farmers cannot act quickly |

---

## ✨ Features

### 🔬 Seven-Step Detection Pipeline
Every uploaded image goes through a complete automated pipeline before results are shown.

### 🌐 Bilingual Interface (English + Urdu)
All user-facing text appears in **both languages simultaneously** — no toggle required. Urdu uses right-to-left (RTL) layout throughout.

### 🔐 User Authentication
- **Email signup** with Firebase email/password and mandatory email verification
- **Phone OTP signup** via Firebase Authentication
- **Forgot password** — sends a Firebase password reset email
- **Guest mode** — full analysis access without account creation

### 📜 Upload History
Authenticated users have every scan saved to Firebase Firestore and displayed in the sidebar, sorted by most recent.

### 📊 Confidence Thresholding
If model confidence is below **80%**, the system refuses to show a disease label and asks for a clearer image — preventing misleading predictions.

### 🔴 Severity Assessment
Based on confidence score and brown pixel ratio, disease severity is classified as **Mild**, **Moderate**, or **Severe**.

### 💊 Treatment & Prevention Advice
Disease-specific, actionable treatment steps and prevention tips are shown in both English and Urdu for all three disease classes.

---

## 🔬 Detection Pipeline

```
📷 Image Quality Check
        ↓
🌿 Leaf Detection Check
        ↓
🧠 CNN Disease Detection
        ↓
📊 Confidence Score (80% threshold)
        ↓
🔴 Severity Assessment
        ↓
💊 Treatment Advice
        ↓
🛡️ Prevention Tips
```

| Step | What it does |
|---|---|
| **Image Quality Check** | Validates size (min 100×100px), brightness (30–245). Rejects poor-quality images. |
| **Leaf Detection** | Analyses green/brown pixel ratio to verify a leaf is present. |
| **CNN Detection** | Resizes image to 256×256, normalises, and runs inference through the Keras model. |
| **Confidence Score** | Displays prediction probability as a colour-coded progress bar. |
| **Confidence Threshold** | If confidence < 80%, shows "Unable to Confidently Identify" instead of a label. |
| **Severity** | Estimates Mild / Moderate / Severe based on confidence and brown pixel ratio. |
| **Treatment + Prevention** | Disease-specific advice in English and Urdu. |

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **ML Framework** | TensorFlow 2.15 / Keras |
| **Dataset** | PlantVillage — 2,152 labeled potato leaf images |
| **Web Framework** | Streamlit (Python) |
| **Authentication** | Firebase Authentication (REST API) |
| **Database** | Firebase Firestore |
| **Hosting** | Streamlit Cloud |
| **Version Control** | GitHub |
| **Language** | Python 3.11 |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | Custom CNN — 3× Conv2D + MaxPool, Flatten, Dense(64), Softmax(3) |
| **Input Size** | 256 × 256 × 3 (RGB) |
| **Classes** | `["Early Blight", "Late Blight", "Healthy"]` |
| **Training Epochs** | 20 |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Cross-Entropy |
| **Data Augmentation** | RandomFlip (horizontal + vertical), RandomRotation (0.2) |
| **Training Accuracy** | **97.22%** |
| **Validation Accuracy** | **94.79%** |
| **Training Loss** | 0.0798 |
| **Validation Loss** | 0.1266 |
| **Saved Format** | `potato_disease_model.keras` |

### CNN Architecture

```
Input (256×256×3)
    → Rescaling (normalize to [0,1])
    → RandomFlip + RandomRotation (augmentation)
    → Conv2D(32, 3×3, ReLU) → MaxPooling2D
    → Conv2D(64, 3×3, ReLU) → MaxPooling2D
    → Conv2D(64, 3×3, ReLU) → MaxPooling2D
    → Flatten
    → Dense(64, ReLU)
    → Dense(3, Softmax)  ← Output
```

> ⚠️ **Important:** The class order `["Early Blight", "Late Blight", "Healthy"]` matches the alphabetical order of the PlantVillage dataset folders (`Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`) and must not be changed.

---

## 📁 Project Structure

```
Potato-Disease-Detection/
│
├── app.py                      # Main Streamlit application
├── potato_disease_model.keras  # Trained CNN model
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version (python-3.11)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
└── Model-Training.ipynb        # Jupyter notebook — model training
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11
- Git
- A Firebase project with:
  - Authentication enabled (Email/Password + Phone)
  - Firestore database created
  - Web API key

### 1. Clone the repository

```bash
git clone https://github.com/Fatima-3015/Potato-Disease-Detection.git
cd Potato-Disease-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Firebase secrets

Create a `.streamlit/secrets.toml` file:

```toml
firebase_web_api_key = "YOUR_FIREBASE_WEB_API_KEY"

[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

> ⚠️ **Never commit** `.streamlit/secrets.toml` to GitHub. It is listed in `.gitignore`.

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deployment

The app is deployed on **Streamlit Cloud** and auto-deploys whenever code is pushed to the `main` branch.

### Deploy your own fork

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → **Deploy a public app from GitHub**
4. Select your repository, branch `main`, and main file `app.py`
5. Add your Firebase secrets in the **Secrets** section of the app settings
6. Click **Deploy**

### Python version

The `runtime.txt` file pins Python 3.11 for Streamlit Cloud compatibility with TensorFlow:

```
python-3.11
```

### Dependencies (`requirements.txt`)

```
streamlit
tensorflow==2.15.0
pillow
numpy
```

---

## 🔐 Authentication

### Email Authentication
- Uses **Firebase Authentication REST API**
- Signup requires email verification before login is allowed
- Forgot password sends a Firebase password reset email

### Phone Authentication (Demo Mode)
- Uses **Firebase phone OTP** via JavaScript SDK embedded in the app
- Demo credentials for testing:
  - **Phone:** `+923001234567`
  - **OTP:** `123456`
- Real SMS requires a Firebase Blaze (paid) plan

### Guest Mode
- No account required
- Full analysis pipeline available
- Scan history is **not** saved

---

## 👥 Team

| Name | Role |
|---|---|
| **Fatima Mahmood** | Team Leader |
| **Momina Hussain** | Team Member |
| **Syeda Maryam Bukhari** | Team Member |
| **Laiba Mahmood** | Team Member |
| **Rimsha Sehzadi** | Team Member |
| **Muqaddas** | Team Member |

> Submitted for **Pak Angels Generative & Agentic AI Training — Cohort 11**  
> Delivered by: PAKANGELS | iCode Guru | ASPIRE Pakistan

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| **Domain Shift** | Model trained on PlantVillage (lab images). Performance on real-world field photos with different backgrounds is reduced. |
| **Three Classes Only** | Detects Early Blight, Late Blight, and Healthy only. Other diseases return an uncertain result. |
| **CPU Only** | TensorFlow ≥ 2.11 does not support GPU on native Windows. Inference runs on CPU. |
| **Phone OTP (Demo)** | Real SMS delivery requires Firebase Blaze plan. |
| **No Image Storage** | Uploaded images are processed in memory only — not stored or logged. |

---

## 🚀 Future Work

- [ ] Transfer Learning using VGG16 or ResNet50 for better real-world accuracy
- [ ] Expand to more potato disease classes (Blackleg, Verticillium Wilt)
- [ ] Mobile app (React Native / Flutter) with offline on-device inference
- [ ] GPS-based disease mapping and regional monitoring
- [ ] Real SMS OTP with Firebase Blaze plan
- [ ] Image storage in Firebase Storage for visual scan history
- [ ] Severity-based email/SMS alerts

---

## 📄 License

This project was developed for educational and hackathon purposes.  
© 2026 PotatoCare AI Team — Pak Angels Cohort 11

---

<p align="center">
  <b>🥔 PotatoCare AI — Developed for Pak Angels Hackathon</b><br>
  <a href="https://potatocare-ai.streamlit.app">potatocare-ai.streamlit.app</a>
</p>
