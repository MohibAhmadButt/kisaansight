import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import re
import urllib.parse
import streamlit as st
import requests

try:
    from app.weather import get_crop_weather_advisory, DISTRICT_COORDINATES
    from app.nuskha import generate_kisaan_nuskha
    from app.config import API_BASE_URL
except ModuleNotFoundError:
    from weather import get_crop_weather_advisory, DISTRICT_COORDINATES
    from nuskha import generate_kisaan_nuskha
    from config import API_BASE_URL

st.set_page_config(
    page_title="KisaanSight - AI Agronomy Assistant",
    page_icon="🌱",
    layout="wide"
)

lang = st.sidebar.radio("🌐 Language / زبان منتخب کریں:", ["اردو (Urdu)", "English"])
is_urdu = lang.startswith("اردو")

T = {
    "title": "🌱 KisaanSight (کسان سائٹ)" if is_urdu else "🌱 KisaanSight",
    "sub": "AI پر مبنی کلینیکل زرعی معائنہ، وائس معاون اور اسپرے کیلکولیٹر" if is_urdu else "Clinical Multimodal Agronomy Decision Support System",
    "district_hdr": "📍 مقام اور لائیو موسم" if is_urdu else "📍 Location & Live Microclimate",
    "auto_loc": "خودکار لوکیشن (Auto GPS/IP)" if is_urdu else "Auto-detect Location (GPS/IP)",
    "district_lbl": "یا مینوئل ضلع منتخب کریں:" if is_urdu else "Or select manual district:",
    "temp": "درجہ حرارت" if is_urdu else "Temperature",
    "hum": "ہوا میں نمی" if is_urdu else "Humidity",
    "wind": "ہوا کی رفتار" if is_urdu else "Wind Speed",
    "snap_title": "📸 پتے کی لائیو تصویر لیں یا اپلوڈ کریں" if is_urdu else "📸 Leaf Inspection Input",
    "camera_lbl": "کیمرے سے تصویر لیں" if is_urdu else "Take Live Photo",
    "upload_lbl": "تصویر اپلوڈ کریں" if is_urdu else "Upload Leaf Image",
    "btn_diagnose": "🔍 معائنہ اور کلینیکل تشخیص کریں" if is_urdu else "🔍 Analyze Leaf & Diagnose",
    "eval_hdr": "🔬 کلینیکل تشخیصی رپورٹ" if is_urdu else "🔬 Clinical Assessment Report",
    "clarify_hdr": "💬 صوتی مکالمہ اور وضاحتی سوال" if is_urdu else "💬 Multi-turn Clarification Loop",
    "mic_lbl": "مائیک کا بٹن دبائیں اور بولیں:" if is_urdu else "Press mic button & speak:",
    "btn_submit_voice": "آواز جمع کروائیں" if is_urdu else "Process Spoken Answer",
    "text_placeholder": "علامات یا سوال ٹائپ کریں..." if is_urdu else "Type symptoms or question here...",
    "btn_submit_text": "جواب بھیجیں" if is_urdu else "Submit Text Response",
    "nuskha_hdr": "📄 ڈیجیٹل کسان نسخہ" if is_urdu else "📄 Digital Agronomy Prescription",
    "btn_dl": "📥 کسان نسخہ ڈاؤن لوڈ کریں" if is_urdu else "📥 Download Prescription Card"
}

st.markdown(f"""
    <style>
    .main-title {{ font-size: 2.3rem; font-weight: 700; color: #2E7D32; text-align: center; }}
    .sub-title {{ font-size: 1.05rem; color: #555; text-align: center; margin-bottom: 1.2rem; }}
    .urdu-card {{
        direction: rtl; text-align: right;
        font-size: 1.15rem; font-weight: 600;
        background-color: #F1F8E9; border: 1px solid #C5E1A5;
        border-radius: 10px; padding: 18px; margin: 12px 0; color: #1B5E20; line-height: 1.8;
    }}
    .clarification-card {{
        direction: rtl; text-align: right;
        background-color: #FFF8E1; border: 1px solid #FFE082;
        border-radius: 10px; padding: 18px; margin: 12px 0; color: #E65100;
    }}
    .warning-card {{
        direction: rtl; text-align: right;
        font-size: 1.15rem; font-weight: 600;
        background-color: #FFEBEE; border: 1px solid #FFCDD2;
        border-radius: 10px; padding: 18px; margin: 12px 0; color: #C62828; line-height: 1.8;
    }}
    .transcription-card {{
        direction: rtl; text-align: right;
        font-size: 1.1rem; background-color: #E3F2FD;
        border: 1px solid #90CAF9; border-radius: 8px;
        padding: 14px; margin: 10px 0; color: #0D47A1; line-height: 1.7;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{T["sub"]}</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR WEATHER -----------------
st.sidebar.markdown(f"### {T['district_hdr']}")
use_auto_loc = st.sidebar.checkbox(T["auto_loc"], value=True)

with st.sidebar:
    if use_auto_loc:
        weather = get_crop_weather_advisory("Rawalpindi / Islamabad")
        st.info(f"📌 {'شناخت شدہ مقام' if is_urdu else 'Detected'}: **{weather.get('city', 'Farm Area')}**")
    else:
        selected_district = st.selectbox(T["district_lbl"], list(DISTRICT_COORDINATES.keys()))
        coords = DISTRICT_COORDINATES[selected_district]
        weather = get_crop_weather_advisory(selected_district, coords["lat"], coords["lon"])

    st.metric(label=f"{T['temp']} (°C)", value=f"{weather.get('temperature', '--')} °C")
    st.metric(label=f"{T['hum']} (%)", value=f"{weather.get('humidity', '--')} %")
    st.metric(label=f"{T['wind']} (km/h)", value=f"{weather.get('wind_speed', '--')} km/h")
    st.metric(label="VPD (Vapour Deficit)", value=f"{weather.get('vpd', '--')} kPa")
    st.caption(f"🔬 **VPD Risk:** {weather.get('vpd_risk', 'Normal')}")

    advisory_text = weather.get('advisory_urdu', '') if is_urdu else weather.get('advisory_en', '')
    if weather.get('safe_to_spray', True):
        st.success(f"✅ {advisory_text}")
    else:
        st.error(f"⚠️ {advisory_text}")

# ----------------- MAIN UI -----------------
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader(T["snap_title"])
    
    crop_options = ["خودکار (Auto)", "ٹماٹر (Tomato)", "آلو (Potato)", "مکئی (Corn)", "مرچ (Pepper)"]
    selected_crop_ui = st.radio("🌾 **فصل منتخب کریں (Select Crop):**", crop_options, horizontal=True)
    
    crop_hint_map = {
        "خودکار (Auto)": "Auto",
        "ٹماٹر (Tomato)": "Tomato",
        "آلو (Potato)": "Potato",
        "مکئی (Corn)": "Corn",
        "مرچ (Pepper)": "Pepper"
    }
    crop_hint_val = crop_hint_map.get(selected_crop_ui, "Auto")

    tab1, tab2 = st.tabs([T["camera_lbl"], T["upload_lbl"]])
    image_file = None
    with tab1:
        camera_pic = st.camera_input("Point camera directly at leaf")
        if camera_pic:
            image_file = camera_pic
    with tab2:
        uploaded_pic = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"])
        if uploaded_pic:
            image_file = uploaded_pic

    if image_file is not None:
        st.image(image_file, caption="Input Leaf Sample", use_container_width=True)
        if st.button(T["btn_diagnose"], type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf with Crop Prior..."):
                try:
                    files = {"file": (image_file.name, image_file.getvalue(), "image/jpeg")}
                    data_form = {"crop_hint": crop_hint_val}
                    response = requests.post(f"{API_BASE_URL}/diagnose/image", files=files, data=data_form)
                    if response.status_code == 200:
                        st.session_state["diagnosis_result"] = response.json()
                        st.session_state.pop("voice_resolution", None)
                    else:
                        st.error(f"Server Error ({response.status_code}): {response.text}")
                except Exception:
                    st.error("⚠️ Backend offline! Start Uvicorn: uvicorn app.main:app --port 8000")

with col_output:
    if "diagnosis_result" in st.session_state:
        data = st.session_state["diagnosis_result"]
        status = data.get("status")
        confidence = data.get("confidence", 0.0)
        session_id = data.get("session_id")
        severity_score = data.get("severity_score", 0.0)

        st.subheader(T["eval_hdr"])

        if status in ["invalid_leaf", "unresolved_out_of_distribution"]:
            st.markdown(f"""
                <div class="warning-card">
                    <b>⚠️ {data.get('disease_urdu', 'Warning')}</b><br>
                    {data.get('remedy_urdu', '')}
                </div>
            """, unsafe_allow_html=True)
            if data.get("audio_url"):
                st.audio(f"{API_BASE_URL}{data['audio_url']}", format="audio/mp3")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.progress(min(confidence, 1.0), text=f"AI Confidence: {confidence * 100:.1f}%")
            with col_m2:
                st.progress(min(severity_score / 100.0, 1.0), text=f"Infection Severity: {severity_score}%")

            local_heatmap_path = None
            if data.get("heatmap_url"):
                st.image(f"{API_BASE_URL}{data['heatmap_url']}", caption="🎯 Grad-CAM Biological Lesion Localization", use_container_width=True)
                local_heatmap_path = os.path.join("static", data["heatmap_url"].replace("/static/", ""))

            if status == "diagnosed":
                st.markdown(f"""
                    <div class="urdu-card">
                        <b>تشخیص:</b> {data.get('disease_urdu', '')}<br>
                        <b>مستند اسپرے فارمولیشن:</b> {data.get('dealer_brand', 'Recommended')}<br>
                        <b>16L ٹینکی خوراک:</b> {data.get('tank_dose', '')}<br>
                        <b>تفصیلی علاج:</b> {data.get('remedy_urdu', '')}
                    </div>
                """, unsafe_allow_html=True)
            elif status == "clarification_needed":
                raw_q = data.get('question_urdu', '')
                # Isolate English words in brackets using BDI tags to prevent sentence reversal
                clean_q = re.sub(r'(\([A-Za-z\s]+\))', r'<bdi dir="ltr" style="font-weight:bold; color:#B71C1C;"> \1 </bdi>', raw_q)

                st.markdown(f"""
                    <div class="clarification-card">
                        <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 8px; border-bottom: 1px solid #FFE082; padding-bottom: 6px;">
                            💬 وضاحتی سوال &nbsp;<span style="font-size:0.85rem; color:#8D6E63; font-weight:normal;">(AI Clarification Loop)</span>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 600; line-height: 1.8;">
                            {clean_q}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            if data.get("audio_url"):
                st.audio(f"{API_BASE_URL}{data['audio_url']}", format="audio/mp3")

            # Clarification Follow-up Dialogue
            if status == "clarification_needed" or "voice_resolution" in st.session_state:
                st.divider()
                st.subheader(T["clarify_hdr"])
                mode_tabs = st.tabs(["⚡ فوری علامات (Quick Chips)", "🎙️ آواز (Voice Mic)", "✍️ لکھیں (Type Text)"])

                with mode_tabs[0]:
                    st.caption("علامات کا انتخاب کریں:")
                    c1, c2, c3 = st.columns(3)
                    quick_pick = None
                    with c1:
                        if st.button("چھوٹے کالے دھبے اور پیلا پن", use_container_width=True):
                            quick_pick = "پتوں پر چھوٹے باریک سیاہ دھبے ہیں اور پیلا پن ہے"
                    with c2:
                        if st.button("خشک گول چھلے دار دائرے", use_container_width=True):
                            quick_pick = "پتوں پر گول چھلے اور خشک براؤن نشان ہیں"
                    with c3:
                        if st.button("پتے اوپر مڑ رہے ہیں", use_container_width=True):
                            quick_pick = "پتے پیلے ہو کر اوپر کی طرف مڑ رہے ہیں"

                    if quick_pick:
                        try:
                            payload = {"text_input": quick_pick, "session_id": session_id or ""}
                            t_resp = requests.post(f"{API_BASE_URL}/diagnose/ask-text", data=payload)
                            if t_resp.status_code == 200:
                                st.session_state["voice_resolution"] = t_resp.json()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                with mode_tabs[1]:
                    recorded_audio = st.audio_input(T["mic_lbl"])
                    if recorded_audio is not None and st.button(T["btn_submit_voice"], type="primary", use_container_width=True):
                        with st.spinner("Whisper transcribing..."):
                            try:
                                voice_payload = {"file": ("farmer_reply.wav", recorded_audio.getvalue(), "audio/wav")}
                                data_payload = {"session_id": session_id} if session_id else {}
                                v_resp = requests.post(f"{API_BASE_URL}/diagnose/answer-voice", files=voice_payload, data=data_payload)
                                if v_resp.status_code == 200:
                                    st.session_state["voice_resolution"] = v_resp.json()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                with mode_tabs[2]:
                    text_query = st.text_input("علامات یا وضاحت لکھیں:", placeholder=T["text_placeholder"])
                    if st.button(T["btn_submit_text"], use_container_width=True) and text_query.strip():
                        try:
                            payload = {"text_input": text_query, "session_id": session_id or ""}
                            t_resp = requests.post(f"{API_BASE_URL}/diagnose/ask-text", data=payload)
                            if t_resp.status_code == 200:
                                st.session_state["voice_resolution"] = t_resp.json()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            # Active Resolution Parameters
            active_remedy = data.get("remedy_urdu", "")
            active_disease = data.get("disease_urdu", data.get("disease", "Unknown"))
            active_tank_dose = data.get("tank_dose", "40 گرام فی 16L ٹینکی")
            active_brand = data.get("dealer_brand", "Recommended Market Product")

            if "voice_resolution" in st.session_state:
                vdata = st.session_state["voice_resolution"]
                active_remedy = vdata.get("remedy_urdu", active_remedy)
                active_disease = vdata.get("disease_urdu", active_disease)
                active_tank_dose = vdata.get("tank_dose", active_tank_dose)
                active_brand = vdata.get("dealer_brand", active_brand)
                severity_score = vdata.get("severity_score", severity_score)

                st.markdown(f"""
                    <div class="transcription-card">
                        <b>کسان کا بیان / ان پٹ:</b> "{vdata.get('transcription', '')}"
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="urdu-card">
                        <b>حتمی تصدیق:</b> {vdata.get('disease_urdu', '')}<br>
                        <b>مستند دوا:</b> {active_brand}<br>
                        <b>ٹینکی خوراک:</b> {active_tank_dose}<br>
                        <b>مکمل علاج:</b> {vdata.get('remedy_urdu', '')}
                    </div>
                """, unsafe_allow_html=True)
                if vdata.get("audio_url"):
                    st.audio(f"{API_BASE_URL}{vdata['audio_url']}", format="audio/mp3")

            # ----------------- ADVANCED KNAPSACK SPRAY CALCULATOR -----------------
            if status == "diagnosed" or "voice_resolution" in st.session_state:
                st.divider()
                st.subheader("⚙️ کلینیکل اسپرے کیلکولیٹر (Knapsack Tank Mixer)")

                dose_match = re.search(r'\d+', str(active_tank_dose))
                unit_dose = int(dose_match.group()) if dose_match else 40
                unit_label = "ملی لیٹر" if ("ملی" in str(active_tank_dose) or "ml" in str(active_tank_dose).lower()) else "گرام"

                calc_col1, calc_col2 = st.columns([1.1, 2.9])

                with calc_col1:
                    num_tanks = st.number_input(
                        "16L ٹینکیوں کی تعداد:" if is_urdu else "Number of 16L Tanks:",
                        min_value=1,
                        max_value=20,
                        value=2,
                        step=1
                    )
                    st.caption("💡 1 ایکڑ کے لیے عموماً 5 سے 6 ٹینکیاں درکار ہوتی ہیں۔")

                total_chemical = unit_dose * num_tanks
                total_water = 16 * num_tanks
                covered_acres = round(num_tanks / 6.0, 2)
                covered_kanals = round(covered_acres * 8, 1)
                est_cost_min = int(total_chemical * 9.5)
                est_cost_max = int(total_chemical * 13.5)

                with calc_col2:
                    row1_c1, row1_c2 = st.columns(2)
                    row2_c1, row2_c2 = st.columns(2)
                    
                    row1_c1.metric(label="🧪 کل درکار دوا", value=f"{total_chemical} {unit_label}")
                    row1_c2.metric(label="💧 درکار صاف پانی", value=f"{total_water} لیٹر")
                    row2_c1.metric(label="🌾 متوقع رقبہ", value=f"~{covered_acres} ایکڑ", delta=f"{covered_kanals} کنال")
                    row2_c2.metric(label="💵 تخمینہ خرچہ", value=f"Rs {est_cost_min}–{est_cost_max}")

                st.markdown(f"""
                    <div style="background-color: #F1F8E9; border-left: 5px solid #2E7D32; padding: 12px 16px; border-radius: 6px; margin: 10px 0; font-size: 0.95rem; color: #1B5E20; line-height: 1.7; direction: rtl; text-align: right;">
                        <b>📋 درست محلول تیار کرنے کا طریقہ:</b><br>
                        1. پہلے 1 لیٹر بالٹی میں <b>{total_chemical} {unit_label} {active_brand}</b> ڈال کر اچھی طرح حل کریں۔<br>
                        2. ٹینکی کو آدھا پانی سے بھریں، پھر تیار شدہ محلول چھان کر شامل کریں۔<br>
                        3. باقی پانی ڈال کر 16 لیٹر نشان تک بھریں۔ اسپرے نوزل کو باریک فوارے (Cone Nozzle) پر رکھیں۔
                    </div>
                """, unsafe_allow_html=True)

                st.caption("📌 *خوراک عمومی زرعی رہنما اصولوں پر مبنی ہے — کھیت کے مخصوص حالات اور حتمی مقدار کے لیے مقامی زرعی توسیعی عملے سے تصدیق کریں۔*")

                clean_crop = data.get("crop", "فصل")
                msg_body = (
                    f"السلام علیکم!\n"
                    f"میری {clean_crop} کی فصل پر {active_disease} کی تشخیص ہوئی ہے۔\n\n"
                    f"*KisaanSight AI تصدیق شدہ آرڈر:*\n"
                    f"• تجویز کردہ دوا: {active_brand}\n"
                    f"• درکار کل مقدار: {total_chemical} {unit_label} (برائے {num_tanks} ٹینکی)\n"
                    f"• فی ٹینکی خوراک: {active_tank_dose}\n\n"
                    f"کیا آپ کے پاس یہ پروڈکٹ دستیاب ہے؟ ریٹ بتائیں۔"
                )
                encoded_msg = urllib.parse.quote(msg_body)
                wa_url = f"https://wa.me/?text={encoded_msg}"

                st.markdown(f"""
                    <a href="{wa_url}" target="_blank" style="text-decoration: none;">
                        <button style="width: 100%; background: linear-gradient(90deg, #25D366, #128C7E); color: white; border: none; padding: 12px; font-size: 1.05rem; border-radius: 8px; cursor: pointer; font-weight: bold; margin: 8px 0 16px 0; display: flex; align-items: center; justify-content: center; gap: 10px;">
                            📲 ڈیلر کو واٹس ایپ پر آرڈر بھیجیں | {total_chemical} {unit_label} برائے {num_tanks} ٹینکی
                        </button>
                    </a>
                """, unsafe_allow_html=True)

                # Digital Prescription Card Output
                st.subheader(T["nuskha_hdr"])
                nuskha_file = generate_kisaan_nuskha(
                    crop_name=data.get("crop", "Crop"),
                    disease_name=data.get("disease", active_disease),
                    confidence=confidence,
                    remedy_urdu=active_remedy,
                    tank_dose=active_tank_dose,
                    dealer_brand=active_brand,
                    severity_score=severity_score,
                    weather_info=weather,
                    heatmap_img_path=local_heatmap_path if (local_heatmap_path and os.path.exists(local_heatmap_path)) else None
                )
                if os.path.exists(nuskha_file):
                    st.image(nuskha_file, caption="Official Digital Prescription Card", use_container_width=True)
                    with open(nuskha_file, "rb") as card_b:
                        st.download_button(
                            label=T["btn_dl"],
                            data=card_b,
                            file_name="Kisaan_Nuskha.png",
                            mime="image/png",
                            use_container_width=True
                        )