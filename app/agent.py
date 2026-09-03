import uuid
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.config import (
        CONFIDENCE_CLARIFICATION_THRESHOLD,
        MARGIN_CLARIFICATION_THRESHOLD,
        KEYWORD_MATCH_WEIGHT
    )
    from app.diagnoses import (
        DIAGNOSES,
        VOCABULARY_WEIGHTS,
        PAIR_QUESTIONS,
        GENERIC_FALLBACK_QUESTION
    )
except ModuleNotFoundError:
    from config import (
        CONFIDENCE_CLARIFICATION_THRESHOLD,
        MARGIN_CLARIFICATION_THRESHOLD,
        KEYWORD_MATCH_WEIGHT
    )
    from diagnoses import (
        DIAGNOSES,
        VOCABULARY_WEIGHTS,
        PAIR_QUESTIONS,
        GENERIC_FALLBACK_QUESTION
    )

DIAGNOSTIC_SESSIONS = {}

def get_question_for_pair(c1: str, c2: str | None) -> str:
    if not c2:
        return GENERIC_FALLBACK_QUESTION
    return PAIR_QUESTIONS.get(frozenset({c1, c2}), GENERIC_FALLBACK_QUESTION)

def evaluate_crop_state(prediction: dict) -> dict:
    if not prediction.get("valid_leaf", True):
        return {
            "session_id": None,
            "crop": "Unknown",
            "status": "invalid_leaf",
            "disease_urdu": "ناقابلِ شناخت نمونہ (Invalid Leaf)",
            "remedy_urdu": "تصویر میں پودے کا پتہ واضح نظر نہیں آ رہا۔ براہ کرم صرف پتے کی صاف تصویر لیں۔",
            "tank_dose": "N/A",
            "dealer_brand": "N/A",
            "active_ingredient": "None",
            "question_urdu": None,
            "predicted_candidate": None,
            "second_candidate": None,
            "margin": 0.0,
            "confidence": 0.0,
            "severity_score": 0.0
        }

    disease = prediction["disease"]
    confidence = prediction["confidence"]
    crop = prediction.get("crop", "Crop")
    margin = prediction.get("margin", 1.0)
    cand2 = prediction.get("second_candidate")
    severity = prediction.get("severity_score", 0.0)

    if disease == "Unknown_Pathogen_Out_Of_Distribution":
        return {
            "session_id": None,
            "crop": crop,
            "status": "unresolved_out_of_distribution",
            "disease_urdu": "غیر معینہ بیماری (Out of Distribution)",
            "remedy_urdu": "یہ علامات تربیت یافتہ 14 بیماریوں سے مماثلت نہیں رکھتیں۔ غلط دوا کے استعمال سے بچنے کے لیے قریبی زراعت ایکسٹینشن دفتر سے لیبارٹری معائنہ کروائیں۔",
            "tank_dose": "کوئی دوا تجویز نہیں",
            "dealer_brand": "لیبارٹری معائنہ درکار",
            "active_ingredient": "None",
            "question_urdu": None,
            "predicted_candidate": None,
            "second_candidate": None,
            "margin": 0.0,
            "confidence": confidence,
            "severity_score": 0.0
        }

    needs_clarification = (
        confidence < CONFIDENCE_CLARIFICATION_THRESHOLD
        or margin < MARGIN_CLARIFICATION_THRESHOLD
    ) and "healthy" not in disease

    if needs_clarification:
        session_id = str(uuid.uuid4())
        question = get_question_for_pair(disease, cand2)

        DIAGNOSTIC_SESSIONS[session_id] = {
            "crop": crop,
            "top_candidate": disease,
            "second_candidate": cand2,
            "confidence": confidence,
            "severity_score": severity
        }

        return {
            "session_id": session_id,
            "crop": crop,
            "status": "clarification_needed",
            "disease_urdu": None,
            "remedy_urdu": None,
            "tank_dose": None,
            "dealer_brand": None,
            "active_ingredient": None,
            "question_urdu": question,
            "predicted_candidate": disease,
            "second_candidate": cand2,
            "margin": margin,
            "confidence": confidence,
            "severity_score": severity
        }

    info = DIAGNOSES.get(disease, {
        "urdu_name": disease,
        "active_ingredient": "General",
        "tank_dose": "زرعی ماہر کے مطابق",
        "dealer_formulation": "Generic",
        "remedy": "قریبی زرعی ماہر سے مشورہ کریں۔"
    })

    return {
        "session_id": None,
        "crop": crop,
        "status": "diagnosed",
        "disease_urdu": info["urdu_name"],
        "remedy_urdu": info["remedy"],
        "tank_dose": info["tank_dose"],
        "dealer_brand": info["dealer_formulation"],
        "active_ingredient": info["active_ingredient"],
        "question_urdu": None,
        "predicted_candidate": disease,
        "second_candidate": cand2,
        "margin": margin,
        "confidence": confidence,
        "severity_score": severity
    }

def resolve_farmer_reply(session_id: str, transcription: str) -> dict:
    session = DIAGNOSTIC_SESSIONS.get(session_id)
    text = (transcription or "").lower()

    if session:
        top_cand = session.get("top_candidate")
        second_cand = session.get("second_candidate")
        severity = session.get("severity_score", 0.0)
    else:
        top_cand = "Tomato___Bacterial_spot"
        second_cand = "Tomato___Early_blight"
        severity = 0.0

    candidates = [c for c in [top_cand, second_cand] if c]
    scores = {c: 0 for c in candidates}

    for cand in scores.keys():
        weights = VOCABULARY_WEIGHTS.get(cand, [])
        for term in weights:
            if term in text:
                scores[cand] += KEYWORD_MATCH_WEIGHT

    best_candidate = max(scores, key=scores.get) if scores else top_cand
    if not scores or scores[best_candidate] == 0:
        best_candidate = top_cand

    info = DIAGNOSES.get(best_candidate, DIAGNOSES["Tomato___Bacterial_spot"])
    DIAGNOSTIC_SESSIONS.pop(session_id, None)

    return {
        "resolved_disease": best_candidate,
        "disease_urdu": info["urdu_name"],
        "remedy_urdu": info["remedy"],
        "tank_dose": info["tank_dose"],
        "dealer_brand": info["dealer_formulation"],
        "active_ingredient": info["active_ingredient"],
        "severity_score": severity
    }