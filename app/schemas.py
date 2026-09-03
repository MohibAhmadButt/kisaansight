from typing import Optional
from pydantic import BaseModel

class DiagnosisResponse(BaseModel):
    session_id: Optional[str] = None
    crop: str
    status: str
    disease_urdu: Optional[str] = None
    remedy_urdu: Optional[str] = None
    tank_dose: Optional[str] = None
    dealer_brand: Optional[str] = None
    active_ingredient: Optional[str] = None
    question_urdu: Optional[str] = None
    predicted_candidate: Optional[str] = None
    second_candidate: Optional[str] = None
    margin: Optional[float] = 0.0
    confidence: float = 0.0
    severity_score: float = 0.0
    heatmap_url: Optional[str] = None
    audio_url: Optional[str] = None

class ClarificationReplyResponse(BaseModel):
    transcription: str
    resolved_disease: Optional[str] = None
    disease_urdu: str
    remedy_urdu: str
    tank_dose: str
    dealer_brand: str
    active_ingredient: str
    severity_score: float = 0.0
    audio_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    project: str
    docs: str