import pytest
from app.vision import DISEASE_CLASSES
from app.agent import (
    DIAGNOSES,
    VOCABULARY_WEIGHTS,
    evaluate_crop_state,
    resolve_farmer_reply,
    PAIR_QUESTIONS,
    DIAGNOSTIC_SESSIONS
)

def test_every_disease_class_has_a_diagnosis_entry():
    """Verify that all 14 vision classes exist in DIAGNOSES lookup."""
    missing = [c for c in DISEASE_CLASSES if c not in DIAGNOSES]
    assert not missing, f"Classes missing from DIAGNOSES: {missing}"

def test_every_disease_class_has_vocabulary_weights():
    """Verify all 14 classes have vocabulary keywords for clarification matching."""
    missing = [c for c in DISEASE_CLASSES if c not in VOCABULARY_WEIGHTS]
    assert not missing, f"Classes missing from VOCABULARY_WEIGHTS: {missing}"

def test_high_confidence_healthy_leaf_diagnosed_directly():
    """Verify high confidence healthy prediction bypasses clarification loop."""
    mock_prediction = {
        "valid_leaf": True,
        "disease": "Tomato___healthy",
        "crop": "Tomato",
        "confidence": 0.98,
        "second_candidate": "Tomato___Early_blight",
        "second_confidence": 0.01,
        "margin": 0.97,
        "severity_score": 0.0,
        "heatmap_bytes": None
    }
    result = evaluate_crop_state(mock_prediction)
    assert result["status"] == "diagnosed"
    assert "صحت مند" in result["disease_urdu"]

def test_ambiguous_prediction_triggers_pair_specific_question():
    """Verify that ambiguity routes to exact pair question (Bacterial vs Early Blight)."""
    mock_prediction = {
        "valid_leaf": True,
        "disease": "Tomato___Bacterial_spot",
        "crop": "Tomato",
        "confidence": 0.58,
        "second_candidate": "Tomato___Early_blight",
        "second_confidence": 0.42,
        "margin": 0.16,
        "severity_score": 5.2,
        "heatmap_bytes": None
    }
    result = evaluate_crop_state(mock_prediction)
    assert result["status"] == "clarification_needed"
    expected_q = PAIR_QUESTIONS[frozenset({"Tomato___Bacterial_spot", "Tomato___Early_blight"})]
    assert result["question_urdu"] == expected_q

def test_resolve_farmer_reply_matches_keyword_accurately():
    """Verify farmer description resolves to correct candidate."""
    session_id = "test-session-mock"
    DIAGNOSTIC_SESSIONS[session_id] = {
        "crop": "Tomato",
        "top_candidate": "Tomato___Bacterial_spot",
        "second_candidate": "Tomato___Early_blight",
        "confidence": 0.55,
        "severity_score": 6.0
    }
    # Farmer talks about concentric rings (Early Blight symptom)
    result = resolve_farmer_reply(session_id, "پتوں پر گول چھلے اور خشک نشان ہیں")
    assert result["resolved_disease"] == "Tomato___Early_blight"

def test_generic_reply_falls_back_to_top_candidate():
    """Verify generic question without symptoms safely falls back to vision prediction."""
    session_id = "test-session-fallback"
    DIAGNOSTIC_SESSIONS[session_id] = {
        "crop": "Tomato",
        "top_candidate": "Tomato___Bacterial_spot",
        "second_candidate": "Tomato___Early_blight",
        "confidence": 0.65,
        "severity_score": 4.5
    }
    # Generic question without any symptom words
    result = resolve_farmer_reply(session_id, "issue kya hai")
    assert result["resolved_disease"] == "Tomato___Bacterial_spot"