# 🌱 KisaanSight (کسان سائٹ)

### Clinical Multimodal Agronomy Decision Support System (CDSS) for Smallholder Farmers

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-6%20Passed-2E7D32?logo=pytest\&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

**KisaanSight** is an enterprise-grade multimodal **Clinical Agronomy Decision Support System (CDSS)** designed to support smallholder crop pathology workflows across Pakistan.

Rather than treating plant pathology as a conventional multi-class image classification problem, KisaanSight bridges the gap between raw neural-network inference and actionable agronomic intervention through:

1. **Calibrated Computer Vision** — Foliage-relative infection indexing with automated foliage gatekeeping and temperature-scaled Out-of-Distribution (OOD) rejection.
2. **Bilingual Spoken Agentic Dialogue** — Order-independent pairwise clarification using Whisper STT and neural TTS when symptom margins are ambiguous.
3. **Alibaba Cloud Reasoning Tier** — Integration with Alibaba Cloud Model Studio using Qwen-2.5 for multi-turn dialectical agronomy explanations.
4. **Microclimate Biometeorology** — Live Vapor Pressure Deficit (VPD) tracking through Open-Meteo telemetry to estimate environmental fungal-risk windows.
5. **Field-Level Treatment Planning** — Practical 16-liter knapsack sprayer calculations, acre-to-kanal coverage, localized PKR cost ranges, and automated WhatsApp dealer requisitions.

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
│  │ Softmax Temperature Scaling (T=1.3) + Grad-CAM                        │  │
│  └─────────────────────────┬──────────────────────────────────────────────┘  │
│                            │                                                  │
│                            ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/agent.py                                                           │  │
│  │                                                                        │  │
│  │ Margin Evaluator                                                       │  │
│  │ Confidence <85% OR Margin <20%                                        │  │
│  │                         │                                              │  │
│  │              ┌──────────┴──────────┐                                   │  │
│  │              ▼                     ▼                                   │  │
│  │       High Confidence         Ambiguous Prediction                     │  │
│  │              │                     │                                   │  │
│  │              ▼                     ▼                                   │  │
│  │       Direct Diagnosis       Pairwise Clarification                    │  │
│  └────────────────────────────────────┬───────────────────────────────────┘  │
│                                       │                                       │
│  ┌────────────────────────────────────▼───────────────────────────────────┐  │
│  │ app/voice.py                                                           │  │
│  │                                                                        │  │
│  │ Whisper STT  ◄──────────────────────────────► Edge-TTS                │  │
│  │                                                                        │  │
│  │                 Spoken Urdu / Voice Loop                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/weather.py                                                         │  │
│  │                                                                        │  │
│  │ Open-Meteo → VPD Calculation → Environmental Spore-Risk Engine        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/agent.py                                                           │  │
│  │                                                                        │  │
│  │ Alibaba Cloud Model Studio → Qwen-2.5 Reasoning Tier                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ app/nuskha.py                                                          │  │
│  │                                                                        │  │
│  │ Digital Nuskha / Prescription Card Generation                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Engineering Innovations

### 1. Vision Reliability & Foliage Gatekeeper

KisaanSight introduces multiple reliability controls before a plant image reaches the treatment recommendation layer.

#### MobileNetV3 Backbone

The vision model is fine-tuned on foliar pathologies spanning:

* 🍅 Tomato
* 🥔 Potato
* 🌽 Corn
* 🫑 Bell Pepper

Model weights are stored in:

```text
models/mobilenet_plant.pth
```

#### Foliage Gatekeeper

The system analyzes HSV green-hue pixel saturation and applies a foliage-ratio threshold:

```text
Foliage Ratio > 0.06
```

Non-leaf inputs such as hands, soil, and unrelated backgrounds are rejected before disease classification.

#### Calibrated Softmax

Temperature scaling is applied using:

```text
T = 1.3
```

This reduces excessive model confidence on uncertain or unseen inputs.

Predictions with calibrated confidence below:

```text
35%
```

are routed toward Out-of-Distribution (OOD) handling.

#### Grad-CAM Explainability

Grad-CAM generates real-time activation overlays that highlight foliar lesion regions associated with the model's prediction.

---

### 2. Multi-Turn Agentic Clarification Loop

When the probability difference between competing disease candidates is small, KisaanSight does not immediately release chemical guidance.

The ambiguity trigger is:

```text
Prediction Margin < 0.20
```

The system generates an order-independent pairwise differential question.

Example:

```text
Tomato Early Blight
        vs.
Bacterial Spot
```

Farmers can respond through:

* Spoken Urdu
* Natural language
* Interactive symptom chips

A domain-weighted vocabulary matcher evaluates the farmer's response before the system proceeds.

---

### 3. Alibaba Cloud Reasoning Tier

KisaanSight integrates **Alibaba Cloud Model Studio** as an additional reasoning layer.

The system uses:

```text
Model: Qwen-2.5
Endpoint: DashScope OpenAI-Compatible API
Model: qwen-plus
```

The reasoning tier enriches the agronomic advisory workflow with localized, multi-turn explanations, including Urdu-oriented responses.

---

### 4. Knapsack Sprayer Dosage Pharmacology

Standard agricultural recommendations frequently assume commercial-scale spraying equipment.

KisaanSight translates recommendations into practical smallholder field units.

#### 16-Liter Knapsack Tanks

Calculates chemical dilution per standard tank.

Example:

```text
40 g / 16 L tank
```

#### Area Conversion

```text
6 tanks ≈ 1 Acre
1 Acre = 8 Kanals
```

#### Local Pricing

The system supports localized PKR cost modeling using prevalent distributor formulations, including examples such as:

* Cuprocaffaro
* Kocide
* Ridomil Gold

#### WhatsApp Ordering

Generates a pre-filled requisition message for communication with regional agrochemical dealers.

---

## 🌦️ Microclimate & VPD Risk Engine

KisaanSight integrates **Open-Meteo** telemetry to provide environmental context for crop disease risk.

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

This allows visual disease predictions to be considered alongside local microclimate conditions.

---

## 🗣️ Bilingual Voice Interaction

The voice layer enables farmers to interact with the system using spoken language.

```text
Farmer Speech
      │
      ▼
   Whisper
      │
      ▼
Speech-to-Text
      │
      ▼
Agentic Clarification
      │
      ▼
Agronomic Response
      │
      ▼
   Edge-TTS
      │
      ▼
Spoken Response
```

This architecture is intended to reduce dependence on text-heavy interfaces and support Urdu-oriented field interaction.

---

## 🧪 Automated Pipeline Verification

The core clinical decision rules, vocabulary matrix, ambiguity handling, and fallback routing are validated through deterministic `pytest` tests.

Run the test suite with:

```bash
python -m pytest tests/test_pipeline.py -v
```

### Verified Test Output

```text
============================= test session starts ==============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0

tests/test_pipeline.py::test_every_disease_class_has_a_diagnosis_entry PASSED
tests/test_pipeline.py::test_every_disease_class_has_vocabulary_weights PASSED
tests/test_pipeline.py::test_high_confidence_healthy_leaf_diagnosed_directly PASSED
tests/test_pipeline.py::test_ambiguous_prediction_triggers_pair_specific_question PASSED
tests/test_pipeline.py::test_resolve_farmer_reply_matches_keyword_accurately PASSED
tests/test_pipeline.py::test_generic_reply_falls_back_to_top_candidate PASSED

============================== 6 passed in 5.22s ===============================
```

### Verified Behaviors

The test suite validates:

* Disease-class diagnosis mappings
* Disease vocabulary weights
* High-confidence direct diagnosis
* Ambiguous prediction detection
* Pair-specific clarification generation
* Farmer response matching
* Generic-response fallback behavior

---

## 🚀 Quickstart & Installation

### Prerequisites

* Python 3.10–3.12
* Git
* `pip`
* Virtual environment (`venv` or `conda`)

---

### 1. Clone the Repository

```bash
git clone https://github.com/MohibAhmadButt/kisaansight.git

cd kisaansight
```

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Alibaba Cloud Model Studio is optional.

Copy the example environment file:

```bash
cp .env.example .env
```

Then configure:

```env
ALIBABA_CLOUD_API_KEY=your_dashscope_api_key_here
KISAANSIGHT_API_URL=http://127.0.0.1:8000
```

> If no Alibaba Cloud API key is provided, the system falls back to deterministic offline clinical rules without interrupting the core workflow.

---

## ▶️ Running the Application

KisaanSight uses a FastAPI backend and Streamlit frontend.

### Start the FastAPI Backend

```bash
uvicorn app.main:app --port 8000 --reload
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit Frontend

Open a second terminal and run:

```bash
streamlit run app/ui.py
```

Frontend:

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
│   │   └── Clinical state evaluator, ambiguity detection & Alibaba Qwen tier
│   │
│   ├── config.py
│   │   └── Centralized thresholds, parameters & constants
│   │
│   ├── diagnoses.py
│   │   └── Agronomic pathology database & vocabulary weights
│   │
│   ├── main.py
│   │   └── FastAPI backend with Pydantic schemas
│   │
│   ├── nuskha.py
│   │   └── Digital prescription / Nuskha card generator
│   │
│   ├── schemas.py
│   │   └── Strict Pydantic response models
│   │
│   ├── ui.py
│   │   └── Multimodal Streamlit clinical web application
│   │
│   ├── vision.py
│   │   └── MobileNetV3 classifier & Grad-CAM visualizer
│   │
│   ├── voice.py
│   │   └── Whisper STT & Edge-TTS synthesis engines
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
│       └── Deterministic end-to-end unit test suite
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧩 Component Overview

| Component      | Responsibility                                                                 |
| -------------- | ------------------------------------------------------------------------------ |
| `vision.py`    | Plant image classification, foliage validation, calibration, and Grad-CAM      |
| `agent.py`     | Diagnostic confidence evaluation, clarification logic, and Qwen reasoning tier |
| `diagnoses.py` | Agronomic disease database and vocabulary mappings                             |
| `weather.py`   | Open-Meteo integration and VPD risk analysis                                   |
| `voice.py`     | Whisper speech recognition and Edge-TTS synthesis                              |
| `nuskha.py`    | Digital Nuskha / prescription card generation                                  |
| `schemas.py`   | Strict API response validation                                                 |
| `main.py`      | FastAPI application and REST API                                               |
| `ui.py`        | Streamlit user interface                                                       |
| `config.py`    | Centralized configuration and thresholds                                       |

---

## 🔄 End-to-End Workflow

```text
                    Farmer
                      │
              ┌───────┴────────┐
              │                │
          Leaf Photo       Voice Input
              │                │
              └───────┬────────┘
                      ▼
             Foliage Validation
                      │
                      ▼
                MobileNetV3
                      │
                      ▼
             Confidence Calibration
                      │
                ┌─────┴─────┐
                │           │
             Certain     Ambiguous
                │           │
                │           ▼
                │     Pairwise Question
                │           │
                │           ▼
                │     Farmer Response
                │           │
                └─────┬─────┘
                      │
                      ▼
             Agronomic Diagnosis
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     VPD / Weather          Qwen-2.5 Reasoning
       Context                    │
          │                       │
          └───────────┬───────────┘
                      ▼
             Treatment Calculation
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     16L Tank / Acre        PKR Cost Estimate
          │                       │
          └───────────┬───────────┘
                      ▼
              Digital Nuskha
                      │
                      ▼
             WhatsApp Requisition
```

---

## 🌍 Built for Smallholder Agriculture

KisaanSight is designed around practical field conditions and localized agricultural workflows.

The system incorporates:

* 🇵🇰 Pakistani agricultural units
* 💧 16-liter knapsack sprayer calculations
* 📐 Acre-to-kanal conversion
* 💰 PKR-oriented cost modeling
* 🗣️ Urdu-friendly voice interaction
* 🌦️ District-based environmental context
* 📱 WhatsApp-based dealer requisitions
* 🌱 Field-level disease clarification

---

## 🔒 Security

Never commit API credentials or secrets to the repository.

Recommended `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

Use environment variables or your deployment platform's secret-management system for production credentials.

---

## ⚠️ Agronomic Safety Disclaimer

> **IMPORTANT NOTICE**
>
> KisaanSight is an experimental agricultural decision-support and educational system.
>
> AI-generated disease classifications, treatment recommendations, dosage calculations, environmental risk estimates, and product information may contain errors.
>
> Chemical application decisions must be verified against the applicable product label, local agricultural regulations, crop-specific requirements, and guidance from a qualified agronomist or agricultural professional.
>
> KisaanSight must not be treated as a substitute for professional agronomic judgment.

---

## 📄 License

Developed for **smart agriculture innovation and smallholder empowerment**.

Distributed under the **MIT License**.

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

If you find **KisaanSight** useful or interesting, consider giving the repository a ⭐ on GitHub.

**Built for smarter, more accessible, and field-oriented agricultural decision support.** 🌱
