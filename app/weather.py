import math
import requests
from typing import Dict, Any, Tuple

DISTRICT_COORDINATES = {
    "Rawalpindi / Islamabad": {"lat": 33.5973, "lon": 73.0479},
    "Faisalabad": {"lat": 31.4187, "lon": 73.0791},
    "Multan": {"lat": 30.1575, "lon": 71.5249},
    "Okara / Sahiwal": {"lat": 30.8138, "lon": 73.4534},
    "Rahim Yar Khan": {"lat": 28.4212, "lon": 70.2989},
    "Peshawar": {"lat": 34.0151, "lon": 71.5249},
    "Hyderabad": {"lat": 25.3960, "lon": 68.3578}
}

def get_realtime_location() -> Tuple[str, float, float]:
    """Fetches user approximate city and coordinates via IP."""
    try:
        res = requests.get("https://ipapi.co/json/", timeout=3)
        if res.status_code == 200:
            data = res.json()
            city = data.get("city") or "Rawalpindi"
            lat = float(data.get("latitude", 33.5973))
            lon = float(data.get("longitude", 73.0479))
            return city, lat, lon
    except Exception:
        pass
    return "Rawalpindi / Islamabad", 33.5973, 73.0479

def calculate_vpd(temp_c: float, relative_humidity: float) -> float:
    """Calculates Vapour Pressure Deficit (VPD in kPa)."""
    # Saturated vapour pressure
    svp = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual vapour pressure
    avp = svp * (relative_humidity / 100.0)
    vpd = svp - avp
    return round(vpd, 2)

def get_crop_weather_advisory(district_name: str = "Rawalpindi / Islamabad", lat: float = None, lon: float = None) -> Dict[str, Any]:
    city_name = district_name

    if lat is None or lon is None:
        if district_name in DISTRICT_COORDINATES:
            lat = DISTRICT_COORDINATES[district_name]["lat"]
            lon = DISTRICT_COORDINATES[district_name]["lon"]
        else:
            detected_city, lat, lon = get_realtime_location()
            city_name = detected_city

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&"
        f"forecast_days=1"
    )
    
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        current = data.get("current", {})
        temp = float(current.get("temperature_2m", 28.0))
        humidity = float(current.get("relative_humidity_2m", 60.0))
        rain = float(current.get("precipitation", 0.0))
        wind = float(current.get("wind_speed_10m", 5.0))
        
        vpd = calculate_vpd(temp, humidity)
        
        safe_to_spray = True
        warning_urdu = "موسم سپرے کے لیے سازگار ہے۔"
        warning_en = "Optimal weather for spray. No rain, wind & temperature within limits."
        
        # VPD Spore Germination Risk Logic
        vpd_risk = "Normal"
        if vpd < 0.45:
            vpd_risk = "High Spore Germination Risk (نم ہوا - فنگس کا پھیلاؤ تیز)"
        elif vpd > 1.8:
            vpd_risk = "High Transpiration / Stress (خشک گرم ہوا)"

        if rain > 0.1 or humidity > 85:
            safe_to_spray = False
            warning_urdu = "بارش کا امکان یا نمی 85 فیصد سے زیادہ ہے۔ سپرے نہ کریں، دوا بہہ جائے گی۔"
            warning_en = "High rain risk or humidity (>85%). Avoid spraying."
        elif wind > 18:
            safe_to_spray = False
            warning_urdu = "تیز ہوا چل رہی ہے۔ سپرے اڑ کر ضائع ہو جائے گا۔"
            warning_en = "High wind speed (>18 km/h). Spray will drift."
        elif temp > 35:
            warning_urdu = "شدید گرمی ہے۔ سپرے شام کے وقت کریں تاکہ پتے نہ جھلسیں۔"
            warning_en = "High temperature (>35°C). Spray during evening hours."
            
        return {
            "city": city_name or "Farm Area",
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind,
            "vpd": vpd,
            "vpd_risk": vpd_risk,
            "safe_to_spray": safe_to_spray,
            "advisory_urdu": warning_urdu,
            "advisory_en": warning_en
        }
    except Exception:
        return {
            "city": city_name or "Farm Area",
            "temperature": 29.0,
            "humidity": 55.0,
            "wind_speed": 6.0,
            "vpd": 1.05,
            "vpd_risk": "Normal",
            "safe_to_spray": True,
            "advisory_urdu": "موسم کا لائیو ڈیٹا دستیاب نہیں۔ معمول کے مطابق احتیاط اپنائیں۔",
            "advisory_en": "Weather offline. Follow general agronomy precautions."
        }