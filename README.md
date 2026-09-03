# 🌱 KisaanSight (کسان سائٹ)

### Clinical Agronomy Decision Support System (CDSS) for Smallholder Farmers

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-6%20Passed-2E7D32?logo=pytest\&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

**KisaanSight** is a multimodal **Clinical Agronomy Decision Support System (CDSS)** designed to support smallholder crop pathology workflows across Pakistan.

Rather than treating plant pathology as a conventional image-classification problem, KisaanSight connects computer vision with agentic reasoning, microclimate analysis, voice interaction, and practical field-level intervention guidance.

The system bridges the gap between raw neural-network predictions and actionable agronomic recommendations through four core capabilities:

1. **Calibrated Computer Vision** — Foliage-relative infection indexing, automated foliage validation, temperature-scaled confidence calibration, and Out-of-Distribution (OOD) rejection.
2. **Bilingual Agentic Dialogue** — Order-independent pairwise clarification using Whisper speech recognition and neural text-to-speech when predictions are ambiguous.
3. **Microclimate Biometeorology** — Live Vapor Pressure Deficit (VPD) monitoring to estimate environmental fungal-risk windows.
4. **Field-Level Treatment Planning** — Practical 16-liter knapsack sprayer calculations, acre-to-kanal coverage estimates, localized PKR cost modeling, and WhatsApp dealer requisition generation.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────────────┐
                         │    Farmer / Field Worker     │
                         └──────────────┬───────────────┘
                                        │
                         Leaf Photo / Voice Input
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend — Port 8501                            │
│                                                                               │
│  • Microclimate Dashboard                    • Knapsack Mixer Metrics        │
│  • Auto-GPS / District Selection              • Live Leaf Capture            │
│  • Image Uploader                             • WhatsApp Dealer Order        │
│  • Voice Clarification Loop                   • Digital Nuskha Card          │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                              REST API / JSON
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend — Port 8000                              │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/vision.py                                                          │  │
│  │                                                                        │  │
│  │ Foliage Ratio Gatekeeper (>6% Leaf Check)                              │  │
│  │                         │                                              │  │
│  │                         ▼                                              │  │
│  │ MobileNetV3 — 14 Foliar Pathology Classes                              │  │
│  │                         │                                              │  │
│  │                         ▼                                              │  │
│  │ Temperature Scaling (T=1.3) + Grad-CAM                                │  │
│  └─────────────────────────┬──────────────────────────────────────────────┘  │
│                            │                                                  │
│                            ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/agent.py                                                           │  │
│  │                                                                        │  │
│  │ Margin Evaluator                                                       │  │
│  │ Confidence <85% OR Margin <20%                                        │  │
│  │             │                                                          │  │
│  │       ┌─────┴─────┐                                                    │  │
│  │       ▼           ▼                                                    │  │
│  │ High Confidence   Ambiguous Prediction                                │  │
│  │       │           │                                                    │  │
│  │       ▼           ▼                                                    │  │
│  │ Direct Diagnosis  Pairwise Clarification Question                     │  │
│  └─────────────────────┬──────────────────────────────────────────────────┘  │
│                        │                                                     │
│  ┌─────────────────────▼──────────────────────────────────────────────────┐  │
│  │ app/voice.py                                                          │  │
│  │                                                                        │  │
│  │ Whisper STT  ◄──────────────► Edge-TTS                                │  │
│  │                                                                        │  │
│  │ Spoken Urdu / Bilingual Voice Interaction                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/weather.py                                                        │  │
│  │                                                                        │  │
│  │ Open-Meteo → VPD Calculation → Environmental Spore-Risk Engine        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/nuskha.py                                                         │  │
│  │                                                                        │  │
│  │ Digital Prescription / Nuskha Card Generation                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Technical Pillars

### 1. Vision Reliability & Foliage Gatekeeper

KisaanSight uses a calibrated computer-vision pipeline designed to prevent irrelevant or low-quality inputs from reaching the diagnostic layer.

#### MobileNetV3

A MobileNetV3 backbone is fine-tuned for foliar pathologies spanning:

* Tomato
* Potato
* Corn
* Bell Pepper

#### Foliage Gatekeeper

The system evaluates the green-hue distribution of the input image.

Images failing the foliage threshold are rejected before classification:

```text
Foliage Ratio > 0.06
        │
   ┌────┴────┐
   │         │
  PASS      FAIL
   │         │
   ▼         ▼
Classify    Reject
```

This helps prevent inputs such as hands, soil, and unrelated backgrounds from being interpreted as plant disease.

#### Calibrated Softmax

Temperature scaling with:

```text
T = 1.3
```

is applied to the classifier output to reduce excessive confidence on uncertain inputs.

Predictions with calibrated confidence below:

```text
35%
```

are routed toward Out-of-Distribution (OOD) handling.

#### Grad-CAM Explainability

Grad-CAM overlays provide a visual representation of activation regions associated with the predicted pathology.

---

### 2. Multi-Turn Agentic Clarification Loop

When two candidate diseases have a narrow probability gap, KisaanSight does not immediately release treatment guidance.

The ambiguity trigger is:

```text
Confidence < 85%
OR
Prediction Margin < 20%
```

The system then generates an order-independent pairwise clarification question.

For example:

```text
Tomato Early Blight
        vs.
Bacterial Spot
```

The farmer can respond using:

* Natural language
* Spoken Urdu
* Symptom quick-selection chips

The response is evaluated using domain-weighted diagnostic vocabulary before the system proceeds with agronomic guidance.

---

### 3. Knapsack Sprayer Dosage Calculation

KisaanSight translates field recommendations into practical smallholder application units.

The system supports:

#### 16-Liter Knapsack Tanks

Calculates the required formulation per standard 16-liter tank.

Example:

```text
40 g / 16 L tank
```

#### Area Conversion

Field coverage can be expressed using local agricultural units:

```text
6 tanks ≈ 1 Acre
1 Acre = 8 Kanals
```

#### Local Cost Modeling

The system can model approximate treatment costs in PKR using locally relevant commercial formulations, including examples such as:

* Cuprocaffaro
* Kocide
* Ridomil Gold

---

### 4. Microclimate & VPD Risk Engine

The weather module integrates **Open-Meteo** data to calculate environmental conditions relevant to crop pathology.

The pipeline is:

```text
Weather Data
     │
     ▼
Temperature + Humidity
     │
     ▼
Vapor Pressure Deficit (VPD)
     │
     ▼
Environmental Risk Assessment
     │
     ▼
Fungal Spore-Risk Window
```

This allows the system to combine visual pathology predictions with environmental context.

---

## 🧪 Automated Pipeline Verification

The core decision rules, disease vocabulary matrix, and fallback routing are covered by a deterministic `pytest` test suite.

Run the tests with:

```bash
python -m pytest tests/test_pipeline.py -v
```

### Verified Test Suite

```text
============================= test session starts ==============================
platform win32 -- Python 3.12.4, pytest-9.1.1

tests/test_pipeline.py::test_every_disease_class_has_a_diagnosis_entry PASSED
tests/test_pipeline.py::test_every_disease_class_has_vocabulary_weights PASSED
tests/test_pipeline.py::test_high_confidence_healthy_leaf_diagnosed_directly PASSED
tests/test_pipeline.py::test_ambiguous_prediction_triggers_pair_specific_question PASSED
tests/test_pipeline.py::test_resolve_farmer_reply_matches_keyword_accurately PASSED
tests/test_pipeline.py::test_generic_reply_falls_back_to_top_candidate PASSED

============================== 6 passed in 5.19s ===============================
```

### Test Coverage

The verified pipeline covers:

* Disease-class diagnosis mapping
* Disease vocabulary weights
* High-confidence direct diagnosis
* Ambiguous prediction handling
* Pair-specific clarification questions
* Farmer response resolution
* Generic-response fallback behavior

---

## 🚀 Quickstart

### Prerequisites

* Python 3.10–3.12
* `pip`
* Git
* Virtual environment (`venv` or `conda`)

---

### 1. Clone the Repository

```bash
git clone https://github.com/MohibAhmadButt/kisaansight.git

cd kisaansight
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

KisaanSight uses separate FastAPI and Streamlit processes.

### Start the FastAPI Backend

```bash
uvicorn app.main:app --port 8000 --reload
```

The interactive Swagger API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

### Start the Streamlit Frontend

Open a second terminal and activate the virtual environment.

Then run:

```bash
streamlit run app/ui.py
```

The frontend will be available at:

```text
http://localhost:8501
```

---

## 📂 Repository Structure

```text
kisaansight/
│
├── app/
│   ├── agent.py
│   │   └── Clinical state evaluator & clarification logic
│   │
│   ├── config.py
│   │   └── Centralized thresholds, parameters & constants
│   │
│   ├── diagnoses.py
│   │   └── Agronomic pathology database & vocabulary weights
│   │
│   ├── main.py
│   │   └── FastAPI backend & Pydantic schemas
│   │
│   ├── nuskha.py
│   │   └── Digital prescription / Nuskha card generator
│   │
│   ├── schemas.py
│   │   └── Strict Pydantic response models
│   │
│   ├── ui.py
│   │   └── Multimodal Streamlit web application
│   │
│   ├── vision.py
│   │   └── MobileNetV3 classifier & Grad-CAM visualizer
│   │
│   ├── voice.py
│   │   └── Whisper STT & Edge-TTS engines
│   │
│   └── weather.py
│       └── Open-Meteo microclimate & VPD risk engine
│
├── assets/
│   └── fonts/
│       └── Bundled Unicode typography
│
├── models/
│   └── mobilenet_plant.pth
│       └── Calibrated foliar vision weights
│
├── tests/
│   └── test_pipeline.py
│       └── Deterministic pipeline test suite
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧩 Key Components

| Component      | Responsibility                                                            |
| -------------- | ------------------------------------------------------------------------- |
| `vision.py`    | Plant image classification, foliage validation, calibration, and Grad-CAM |
| `agent.py`     | Diagnostic confidence evaluation and clarification workflow               |
| `diagnoses.py` | Disease knowledge base and vocabulary mappings                            |
| `weather.py`   | Weather retrieval and VPD-based environmental risk analysis               |
| `voice.py`     | Speech-to-text and text-to-speech interaction                             |
| `nuskha.py`    | Digital agronomic prescription card generation                            |
| `schemas.py`   | Strict API response validation                                            |
| `main.py`      | FastAPI application and REST endpoints                                    |
| `ui.py`        | Streamlit user interface                                                  |
| `config.py`    | Centralized system configuration                                          |

---

## 🔄 End-to-End Workflow

```text
Farmer
  │
  ├── Leaf Photo
  │
  └── Voice / Text
        │
        ▼
┌──────────────────────┐
│ Foliage Validation   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ MobileNetV3          │
│ Disease Prediction   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Confidence           │
│ Calibration          │
└──────────┬───────────┘
           │
      ┌────┴────┐
      │         │
   Certain   Ambiguous
      │         │
      │         ▼
      │   Pairwise Question
      │         │
      │         ▼
      │   Farmer Response
      │         │
      └────┬────┘
           │
           ▼
┌──────────────────────┐
│ Agronomic Diagnosis  │
└──────────┬───────────┘
           │
           ├──────────────► VPD / Weather Context
           │
           ▼
┌──────────────────────┐
│ Treatment Calculation│
│ • 16L Tank           │
│ • Acre / Kanal       │
│ • PKR Cost           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Digital Nuskha Card  │
│ / Dealer Order       │
└──────────────────────┘
```

---

## 🌍 Designed for Smallholder Agriculture

KisaanSight focuses on practical agricultural workflows rather than purely theoretical model predictions.

The system is designed around:

* Local Pakistani agricultural units
* Smallholder-scale knapsack spraying
* PKR-oriented cost estimation
* Urdu-friendly voice interaction
* District-based weather context
* Field-level disease clarification
* Actionable agronomic outputs

---

## 🔐 Security & Configuration

Never commit API credentials, private configuration, or other secrets to the repository.

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

For production deployments, configure sensitive values through environment variables or your hosting platform's secret-management system.

---

## ⚠️ Agronomic Safety Disclaimer

> **IMPORTANT NOTICE**
>
> KisaanSight is an experimental agricultural decision-support and educational system.
>
> AI-generated disease classifications, treatment recommendations, dosage calculations, environmental risk estimates, and product information may contain errors.
>
> Chemical application decisions should be verified against the product label, local agricultural extension guidance, applicable regulations, crop-specific requirements, and advice from a qualified agronomist or agricultural professional.
>
> The system must not be treated as a substitute for professional agronomic judgment.

---

## 📄 License

This project is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the full license text.

---

## 👨‍💻 Author

**Mohib Ahmad Butt**

BS Artificial Intelligence
SZABIST Islamabad, Pakistan

**GitHub:**
https://github.com/MohibAhmadButt

**Repository:**
https://github.com/MohibAhmadButt/kisaansight

---

## ⭐ Support

If you find KisaanSight useful or interesting, consider giving the repository a ⭐.

**Built for smarter, more accessible, and field-oriented agricultural decision support.** 🌱
