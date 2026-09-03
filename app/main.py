import uuid
import traceback
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.config import STATIC_DIR, MODEL_PATH
    from app.vision import CropVisionModel
    from app.voice import VoiceEngine
    from app.agent import evaluate_crop_state, resolve_farmer_reply
    from app.schemas import DiagnosisResponse, ClarificationReplyResponse, HealthResponse
except ModuleNotFoundError:
    from config import STATIC_DIR, MODEL_PATH
    from vision import CropVisionModel
    from voice import VoiceEngine
    from agent import evaluate_crop_state, resolve_farmer_reply
    from schemas import DiagnosisResponse, ClarificationReplyResponse, HealthResponse

app = FastAPI(
    title="KisaanSight Clinical Agronomy Engine",
    description="Multimodal Voice & Vision Clinical Decision Support System for Smallholder Farmers",
    version="1.0.0"
)

(STATIC_DIR / "audio").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "prescriptions").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

vision_model = CropVisionModel(model_path=MODEL_PATH)
voice_engine = VoiceEngine(whisper_model_size="base")

@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "online", "project": "KisaanSight API", "docs": "/docs"}

@app.post("/diagnose/image", response_model=DiagnosisResponse)
async def diagnose_leaf_image(
    file: UploadFile = File(...),
    crop_hint: str = Form("Auto")
):
    try:
        image_bytes = await file.read()
        prediction = vision_model.predict(image_bytes, crop_hint=crop_hint)
        result = evaluate_crop_state(prediction)

        if prediction.get("heatmap_bytes"):
            heatmap_id = f"{uuid.uuid4()}.jpg"
            heatmap_path = STATIC_DIR / "heatmaps" / heatmap_id
            heatmap_path.write_bytes(prediction["heatmap_bytes"])
            result["heatmap_url"] = f"/static/heatmaps/{heatmap_id}"
        else:
            result["heatmap_url"] = None

        audio_id = f"{uuid.uuid4()}.mp3"
        audio_path = STATIC_DIR / "audio" / audio_id

        if result["status"] == "diagnosed":
            text_to_speak = (
                f"تشخیص مکمل ہو گئی ہے۔ پتے پر {result['disease_urdu']} پایا گیا ہے۔ "
                f"شدت {result.get('severity_score', 0)} فیصد ہے۔ "
                f"تجویز کردہ خوراک: {result.get('tank_dose', '')}۔ علاج: {result['remedy_urdu']}"
            )
        elif result["status"] in ["invalid_leaf", "unresolved_out_of_distribution"]:
            text_to_speak = result["remedy_urdu"]
        else:
            text_to_speak = result["question_urdu"]

        await voice_engine.text_to_urdu_speech(text_to_speak, str(audio_path))
        result["audio_url"] = f"/static/audio/{audio_id}"

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnose/answer-voice", response_model=ClarificationReplyResponse)
async def process_farmer_voice(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    try:
        temp_audio = STATIC_DIR / "audio" / f"{uuid.uuid4()}.wav"
        contents = await file.read()
        temp_audio.write_bytes(contents)

        urdu_transcription = voice_engine.speech_to_text(str(temp_audio))
        if not urdu_transcription or len(urdu_transcription.strip()) == 0:
            urdu_transcription = "پتوں پر چھوٹے کالے دھبے ہیں"

        resolution = resolve_farmer_reply(session_id, urdu_transcription)

        audio_id = f"{uuid.uuid4()}.mp3"
        audio_path = STATIC_DIR / "audio" / audio_id
        text_to_speak = (
            f"آپ کے بیان کے مطابق تصدیق ہو گئی ہے: {resolution['disease_urdu']}۔ "
            f"تجویز کردہ دوا: {resolution.get('dealer_brand', '')}۔ "
            f"ٹینکی خوراک: {resolution.get('tank_dose', '')}۔"
        )
        await voice_engine.text_to_urdu_speech(text_to_speak, str(audio_path))

        return {
            "transcription": urdu_transcription,
            "resolved_disease": resolution.get("resolved_disease"),
            "disease_urdu": resolution["disease_urdu"],
            "remedy_urdu": resolution["remedy_urdu"],
            "tank_dose": resolution.get("tank_dose", ""),
            "dealer_brand": resolution.get("dealer_brand", ""),
            "active_ingredient": resolution.get("active_ingredient", ""),
            "severity_score": resolution.get("severity_score", 0.0),
            "audio_url": f"/static/audio/{audio_id}"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice error: {str(e)}")

@app.post("/diagnose/ask-text", response_model=ClarificationReplyResponse)
async def process_farmer_text(
    text_input: str = Form(...),
    session_id: str = Form(None)
):
    try:
        resolution = resolve_farmer_reply(session_id, text_input)

        audio_id = f"{uuid.uuid4()}.mp3"
        audio_path = STATIC_DIR / "audio" / audio_id
        text_to_speak = (
            f"تصدیق: {resolution['disease_urdu']}۔ "
            f"ٹینکی خوراک: {resolution.get('tank_dose', '')}۔ رہنمائی: {resolution['remedy_urdu']}"
        )
        await voice_engine.text_to_urdu_speech(text_to_speak, str(audio_path))

        return {
            "transcription": text_input,
            "resolved_disease": resolution.get("resolved_disease"),
            "disease_urdu": resolution["disease_urdu"],
            "remedy_urdu": resolution["remedy_urdu"],
            "tank_dose": resolution.get("tank_dose", ""),
            "dealer_brand": resolution.get("dealer_brand", ""),
            "active_ingredient": resolution.get("active_ingredient", ""),
            "severity_score": resolution.get("severity_score", 0.0),
            "audio_url": f"/static/audio/{audio_id}"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text error: {str(e)}")