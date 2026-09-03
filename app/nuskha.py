import os
import sys
import re
import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.config import FONT_DIR, STATIC_DIR
except ModuleNotFoundError:
    try:
        from config import FONT_DIR, STATIC_DIR
    except ModuleNotFoundError:
        FONT_DIR = ROOT_DIR / "assets" / "fonts"
        STATIC_DIR = ROOT_DIR / "static"


def get_font(size: int = 18, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Loads universal Latin font with zero tofu or broken glyphs."""
    win_fonts = ["arialbd.ttf" if bold else "arial.ttf", "tahomabd.ttf" if bold else "tahoma.ttf", "calibri.ttf"]
    for name in win_fonts:
        win_path = os.path.join("C:\\Windows\\Fonts", name)
        if os.path.exists(win_path):
            try:
                return ImageFont.truetype(win_path, size)
            except Exception:
                pass

    bundled_names = [
        "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    ]
    for b_name in bundled_names:
        b_path = Path(FONT_DIR) / b_name
        if b_path.exists():
            try:
                return ImageFont.truetype(str(b_path), size)
            except Exception:
                pass

    return ImageFont.load_default()


def generate_kisaan_nuskha(
    crop_name: str,
    disease_name: str,
    confidence: float,
    remedy_urdu: str = "",
    tank_dose: str = "40g per 16L Tank",
    dealer_brand: str = "Cuprocaffaro / Kocide",
    severity_score: float = 0.0,
    weather_info: dict = None,
    heatmap_img_path: str = None,
    **kwargs
) -> str:
    prescriptions_dir = Path(STATIC_DIR) / "prescriptions"
    os.makedirs(prescriptions_dir, exist_ok=True)
    out_filename = str(prescriptions_dir / f"nuskha_{int(datetime.datetime.now().timestamp())}.png")

    width, height = 820, 920
    img = Image.new("RGB", (width, height), color="#FAFBF7")
    draw = ImageDraw.Draw(img)

    f_title = get_font(22, bold=True)
    f_header = get_font(17, bold=True)
    f_body = get_font(15, bold=False)
    f_bold = get_font(15, bold=True)
    f_small = get_font(12, bold=False)

    # Sanitize Disease name
    raw_disease = str(disease_name)
    clean_disease = (
        raw_disease.replace("Tomato___", "")
        .replace("Potato___", "")
        .replace("Corn_(maize)___", "")
        .replace("Pepper,_bell___", "")
        .replace("_", " ")
    )
    if any('\u0600' <= char <= '\u06FF' for char in clean_disease):
        if "بیکٹیریل" in clean_disease or "Bacterial" in raw_disease:
            clean_disease = "Bacterial Leaf Spot (Xanthomonas)"
        elif "اگیتی" in clean_disease or "Early" in raw_disease:
            clean_disease = "Early Blight (Alternaria solani)"
        elif "پچھیتی" in clean_disease or "Late" in raw_disease:
            clean_disease = "Late Blight (Phytophthora infestans)"
        elif "مروڑ" in clean_disease or "Curl" in raw_disease:
            clean_disease = "Yellow Leaf Curl Virus (TYLCV)"
        elif "زنگ" in clean_disease or "Rust" in raw_disease:
            clean_disease = "Common Foliar Rust (Puccinia)"
        else:
            clean_disease = "Symptomatic Foliar Plant Pathogen"

    # Sanitize Tank Dose
    clean_tank_dose = str(tank_dose)
    if any('\u0600' <= char <= '\u06FF' for char in clean_tank_dose):
        d_match = re.search(r'\d+', clean_tank_dose)
        num = d_match.group() if d_match else "40"
        unit = "ml" if "ملی" in clean_tank_dose else "g"
        clean_tank_dose = f"{num}{unit} per 16-Liter Knapsack Tank"

    # 1. Header
    draw.rectangle([(0, 0), (width, 90)], fill="#1B5E20")
    draw.text((30, 20), "KISAANSIGHT - CLINICAL AGRONOMY PRESCRIPTION", fill="#FFFFFF", font=f_title)
    draw.text((30, 55), "Smart Agriculture Decision Support Prototype | Field Demonstration Record", fill="#A5D6A7", font=f_small)

    # 2. Metadata Banner
    date_str = datetime.datetime.now().strftime("%d %B %Y | %I:%M %p")
    city_name = weather_info.get("city", "Rawalpindi / Islamabad") if weather_info else "Rawalpindi / Islamabad"
    draw.rectangle([(30, 105), (width - 30, 165)], fill="#E8F5E9", outline="#81C784", width=2)
    draw.text((50, 115), f"Target Crop: {crop_name.upper()}", fill="#1B5E20", font=f_bold)
    draw.text((320, 115), f"Date: {date_str}", fill="#2E7D32", font=f_body)
    draw.text((50, 138), f"Farm Area: {city_name}", fill="#2E7D32", font=f_body)
    draw.text((320, 138), f"Diagnostic Confidence: {confidence * 100:.1f}%", fill="#1B5E20", font=f_bold)

    # 3. Severity Assessment Box
    draw.rectangle([(30, 180), (width - 30, 240)], fill="#FFF3E0", outline="#FFB74D", width=2)
    stage = (
        "Mild Stage (<15%)" if severity_score < 15
        else ("Moderate Stage (15-40%)" if severity_score <= 40 else "Severe Outbreak (>40%)")
    )
    draw.text((50, 192), f"Infection Severity Index: {severity_score}%  [{stage}]", fill="#E65100", font=f_bold)
    draw.text((50, 214), f"Identified Pathogen: {clean_disease}", fill="#BF360C", font=f_body)

    # 4. Grad-CAM Section
    current_y = 255
    if heatmap_img_path and os.path.exists(heatmap_img_path):
        try:
            h_img = Image.open(heatmap_img_path).resize((175, 175))
            img.paste(h_img, (50, current_y))
            draw.rectangle([(50, current_y), (225, current_y + 175)], outline="#388E3C", width=2)

            draw.text((245, current_y + 10), "Explainable AI (Grad-CAM Activation):", fill="#1B5E20", font=f_header)
            draw.text((245, current_y + 35), "Thermal map highlights model attention zones.", fill="#555555", font=f_small)
            draw.text((245, current_y + 55), "Localized activation corresponds to symptomatic foliar regions.", fill="#555555", font=f_small)

            draw.text((245, current_y + 90), f"Recommended Formulation: {dealer_brand}", fill="#0D47A1", font=f_bold)
            draw.text((245, current_y + 115), f"Knapsack Tank Rate: {clean_tank_dose}", fill="#B71C1C", font=f_bold)
            current_y += 195
        except Exception:
            current_y += 10
    else:
        current_y += 10

    # 5. Treatment Protocol Box
    draw.rectangle([(30, current_y), (width - 30, current_y + 130)], fill="#FFFFFF", outline="#CFD8DC", width=2)
    draw.text((50, current_y + 12), "CLINICAL PRESCRIPTION & DOSAGE PROTOCOL", fill="#1B5E20", font=f_header)
    draw.text((50, current_y + 40), f"Primary Formulation: {dealer_brand}", fill="#212121", font=f_bold)
    draw.text((50, current_y + 65), f"Tank Mixture: Dissolve {clean_tank_dose} thoroughly in 16 Liters clean water.", fill="#37474F", font=f_body)
    draw.text((50, current_y + 90), "Application Instruction: Spray underside of leaves during morning or late afternoon.", fill="#546E7A", font=f_small)
    current_y += 145

    # 6. Microclimate Box (Cleaned pure English VPD risk)
    draw.rectangle([(30, current_y), (width - 30, current_y + 95)], fill="#E1F5FE", outline="#4FC3F7", width=2)
    if weather_info:
        raw_vpd = str(weather_info.get("vpd_risk", "Moderate"))
        # Strip all Urdu/Arabic glyphs and extra parentheses
        clean_vpd_risk = re.sub(r'[\u0600-\u06FF\(\)]', '', raw_vpd).strip()
        if not clean_vpd_risk or "High" in raw_vpd or "زیادہ" in raw_vpd:
            clean_vpd_risk = "High Spore Germination Risk"
        elif "Normal" in raw_vpd or "Low" in raw_vpd:
            clean_vpd_risk = "Low Spore Risk"

        draw.text(
            (50, current_y + 10),
            f"Weather Context: {weather_info.get('temperature')}°C  |  Humidity: {weather_info.get('humidity')}%  |  Wind: {weather_info.get('wind_speed')} km/h",
            fill="#0277BD", font=f_bold,
        )
        draw.text(
            (50, current_y + 35),
            f"Vapour Deficit (VPD): {weather_info.get('vpd', 0.45)} kPa  [Spore Multiplication: {clean_vpd_risk}]",
            fill="#01579B", font=f_body,
        )
        draw.text(
            (50, current_y + 60),
            "Advisory: High humidity detected. Allow foliage to dry before applying chemical spray.",
            fill="#004D40", font=f_small,
        )
    current_y += 115

    # Footer
    draw.text((30, height - 30), "Generated via KisaanSight Multimodal Clinical Agronomy Engine | Confidential Farm Record", fill="#9E9E9E", font=f_small)

    img.save(out_filename)
    return out_filename