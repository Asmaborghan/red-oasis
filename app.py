"""
Red Oasis - irrigation decision-support prototype.
NASA Space Apps Challenge 2026.

Run locally:  streamlit run app/app.py

Values that belong to other team members are collected in the
CONFIGURATION block below. Search for "TODO" to find every place
that still needs a real value.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The AI component lives in model/ at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from model.predict import model_status, predict
    AI_IMPORT_ERROR = None
except Exception as exc:  # keep the website working if ML packages are missing
    predict = model_status = None
    AI_IMPORT_ERROR = exc

# ======================================================================
# CONFIGURATION
# ======================================================================

SITE = {
    "name": {"en": "Daris Oasis", "ar": "واحة الدريز"},
    "region": {"en": "Ibri, Ad Dhahirah", "ar": "عبري، الظاهرة"},
    "lat": 23.32,       # TODO (Shareefa): replace with the verified latitude
    "lon": 56.62,       # TODO (Shareefa): replace with the verified longitude
    "verified": False,  # TODO (Shareefa): set True once confirmed on satellite view
    "crop": None,       # TODO (Shareefa): e.g. {"en": "Date palm", "ar": "نخيل التمر"}
}

# Illustrative thresholds for the demonstration advisor.
# TODO (Shareefa): replace with crop-specific values from the research document.
THRESHOLDS = {
    "soil_dry": 30,      # soil moisture (%) below which irrigation is advised
    "soil_moderate": 45,
    "temp_high": 40,     # air temperature (deg C)
    "ndvi_low": 0.30,    # vegetation index below which stress is flagged
}

# TODO (Shareefa): fill in Mars-mode values from NASA closed-environment sources.
MARS_PARAMS = {
    "water": {"en": "Recycled, closed loop", "ar": "إعادة تدوير في دورة مغلقة"},
    "light": {"en": "Artificial", "ar": "صناعية"},
    "temperature": None,
    "crop_water": None,
}

DATA_SOURCES = [
    ("SMAP", "soil", "soil_use"),
    ("MODIS / VIIRS", "ndvi", "ndvi_use"),
    ("GPM IMERG", "rain", "rain_use"),
    ("NASA POWER", "temp", "temp_use"),
]


def get_recommendation(soil: float, temp: float, ndvi: float):
    """Return (level, reasons) for the rule-based demonstration advisor.

    Used only as a fallback when no trained model is available; the
    trained model is called through get_ai_assessment() below.
    """
    score, reasons = 0, []
    if soil < THRESHOLDS["soil_dry"]:
        score += 2
        reasons.append(("r_soil_low", soil))
    elif soil < THRESHOLDS["soil_moderate"]:
        score += 1
        reasons.append(("r_soil_mid", soil))
    if temp >= THRESHOLDS["temp_high"]:
        score += 1
        reasons.append(("r_heat", temp))
    if ndvi < THRESHOLDS["ndvi_low"]:
        score += 1
        reasons.append(("r_veg", ndvi))
    if not reasons:
        reasons.append(("r_ok", None))
    level = "irrigate" if score >= 2 else "monitor" if score == 1 else "none"
    return level, reasons


def get_ai_assessment(soil: float, temp: float, ndvi: float, rain: float):
    """Call Yaqeen's model. Returns the predict() result dict, or None if
    the AI package could not be imported. See model/README.md."""
    if predict is None:
        return None
    return predict(
        {
            "soil_moisture_pct": soil,
            "ndvi": ndvi,
            "rainfall_7d_mm": rain,
            "temperature_c": temp,
        },
        thresholds=THRESHOLDS,
    )


# Model feature name -> TEXT key for its display label.
FEATURE_TEXT = {
    "soil_moisture_pct": "soil",
    "ndvi": "ndvi",
    "rainfall_7d_mm": "rain7",
    "temperature_c": "temp_air",
}


# ======================================================================
# TRANSLATIONS
# ======================================================================

TEXT = {
    "en": {
        "tagline": "From the Falaj to Mars",
        "subtitle": (
            "A decision-support prototype for water-efficient farming, "
            "built on NASA Earth observation data."
        ),
        "badge": "Prototype for the NASA Space Apps Challenge 2026",
        "overview": "Overview",
        "problem_t": "The challenge",
        "problem": (
            "In arid regions, irrigation is often scheduled by habit rather "
            "than by measured crop need, which wastes water or stresses the crop."
        ),
        "approach_t": "Our approach",
        "approach": (
            "We combine satellite measurements of soil moisture, vegetation, "
            "rainfall and weather to estimate crop water stress and advise "
            "when to irrigate."
        ),
        "explain_t": "Explainable decisions",
        "explain": (
            "Every recommendation lists the factors behind it, so farmers "
            "and researchers can check the reasoning."
        ),
        "site": "Test site",
        "site_name": "Site",
        "site_region": "Region",
        "site_crop": "Crop",
        "site_status": "Location status",
        "crop_tbc": "To be confirmed",
        "pending": "Pending verification",
        "verified": "Verified",
        "coords": "Latitude {lat:.5f}, longitude {lon:.5f}",
        "site_note": "Coordinates are provisional until confirmed on satellite imagery.",
        "data": "NASA data sources",
        "c_source": "Source",
        "c_var": "Variable",
        "c_use": "Use in the model",
        "c_status": "Status",
        "planned": "Planned",
        "soil": "Soil moisture",
        "soil_use": "Water availability in the root zone",
        "ndvi": "Vegetation index (NDVI)",
        "ndvi_use": "Crop health and canopy cover",
        "rain": "Precipitation",
        "rain_use": "Rainfall input to the water balance",
        "temp": "Temperature and solar radiation",
        "temp_use": "Evaporative demand",
        "data_note": "The final list will be confirmed against the official challenge statement.",
        "advisor": "Irrigation advisor (demonstration)",
        "advisor_note": (
            "Illustrative inputs and simple rules. This is not the trained "
            "model yet and it does not use live NASA data."
        ),
        "in_soil": "Soil moisture (%)",
        "in_temp": "Air temperature (°C)",
        "in_ndvi": "Vegetation index (NDVI)",
        "result": "Recommendation",
        "irrigate": "Irrigate today",
        "monitor": "Monitor closely",
        "none": "No irrigation needed",
        "why": "Contributing factors",
        "r_soil_low": "Soil moisture is low ({v:.0f}%).",
        "r_soil_mid": "Soil moisture is moderate ({v:.0f}%).",
        "r_heat": "High air temperature increases water demand ({v:.0f} °C).",
        "r_veg": "The vegetation index suggests crop stress (NDVI {v:.2f}).",
        "r_ok": "No stress indicators detected.",
        "in_rain": "Rainfall, last 7 days (mm)",
        "rain7": "Rainfall, last 7 days",
        "temp_air": "Air temperature",
        "ai_note": (
            "Forecast from the trained model. Inputs are entered manually here, "
            "not read live from NASA services."
        ),
        "ai_missing": (
            "The trained model is not available yet, so the simple rules "
            "below are shown instead."
        ),
        "ai_synthetic": (
            "Interface test: this model was trained on synthetic data. The "
            "forecast, recommendation and explanation are not real results."
        ),
        "forecast": "Forecast {label} in {h} days: {v:.1f} {unit}",
        "rec_caveat": (
            "Thresholds are illustrative and SMAP soil moisture is a regional "
            "(~9 km) signal. Confirm with the grower before irrigating."
        ),
        "shap_t": "Why the model gave this forecast (SHAP)",
        "shap_base": "Average model forecast: {v:.1f}",
        "shap_up": "{f} = {x:.2f} raised the forecast by {v:.2f}",
        "shap_down": "{f} = {x:.2f} lowered the forecast by {v:.2f}",
        "shap_imputed": " (missing, filled with a typical value)",
        "shap_note": "These show how the model uses each input. They are not proof of cause.",
        "eval_t": "Model evaluation",
        "eval_line": (
            "Test error (MAE): model {m:.2f}, 'no change' baseline {p:.2f} "
            "on {n} held-out days."
        ),
        "env": "Operating environments",
        "tab_earth": "Earth",
        "tab_mars": "Mars (concept)",
        "earth_text": (
            "Supports irrigation decisions for farms in Oman using "
            "satellite-derived data."
        ),
        "mars_text": (
            "A conceptual extension for closed-environment cultivation, where "
            "water is recycled and resources are limited. It shows how the same "
            "decision logic could be adapted and does not use real Martian "
            "agricultural data."
        ),
        "m_param": "Parameter",
        "m_value": "Value",
        "m_water": "Water supply",
        "m_light": "Lighting",
        "m_temp": "Temperature control",
        "m_crop": "Crop water demand",
        "m_pending": "Pending",
        "footer": (
            "Developed by Team Red Oasis for the NASA Space Apps Challenge 2026. "
            "This is an independent prototype and not an official NASA product. "
            "Data credit: NASA."
        ),
    },
    "ar": {
        "tagline": "من الفلج العُماني إلى المريخ",
        "subtitle": (
            "نموذج أولي لدعم القرار في الزراعة الموفّرة للمياه، "
            "يعتمد على بيانات ناسا لرصد الأرض."
        ),
        "badge": "نموذج أولي لتحدي ناسا لتطبيقات الفضاء 2026",
        "overview": "نظرة عامة",
        "problem_t": "التحدي",
        "problem": (
            "كثيراً ما يُجدوَل الري في المناطق الجافة بالعادة لا بحاجة "
            "المحصول الفعلية، فيحدث هدر للمياه أو إجهاد للنبات."
        ),
        "approach_t": "منهجنا",
        "approach": (
            "ندمج قياسات الأقمار الصناعية لرطوبة التربة والغطاء النباتي "
            "والأمطار والطقس لتقدير الإجهاد المائي للمحصول والتوصية بموعد الري."
        ),
        "explain_t": "قرارات قابلة للتفسير",
        "explain": (
            "تعرض كل توصية العوامل التي استندت إليها، ليتمكن المزارع "
            "والباحث من التحقق من المنطق."
        ),
        "site": "الموقع التجريبي",
        "site_name": "الموقع",
        "site_region": "المنطقة",
        "site_crop": "المحصول",
        "site_status": "حالة الموقع",
        "crop_tbc": "يُحدَّد لاحقاً",
        "pending": "بانتظار التحقق",
        "verified": "تم التحقق",
        "coords": "خط العرض {lat:.5f}، خط الطول {lon:.5f}",
        "site_note": "الإحداثيات مؤقتة إلى أن تُؤكَّد على صور الأقمار الصناعية.",
        "data": "مصادر بيانات ناسا",
        "c_source": "المصدر",
        "c_var": "المتغير",
        "c_use": "الاستخدام في النموذج",
        "c_status": "الحالة",
        "planned": "مخطط",
        "soil": "رطوبة التربة",
        "soil_use": "توفر المياه في منطقة الجذور",
        "ndvi": "مؤشر الغطاء النباتي (NDVI)",
        "ndvi_use": "صحة المحصول وكثافة الغطاء",
        "rain": "الهطول المطري",
        "rain_use": "مدخل الأمطار في الميزان المائي",
        "temp": "الحرارة والإشعاع الشمسي",
        "temp_use": "الطلب التبخري",
        "data_note": "تُؤكَّد القائمة النهائية بحسب نص التحدي الرسمي.",
        "advisor": "مستشار الري (عرض توضيحي)",
        "advisor_note": (
            "مدخلات تجريبية وقواعد مبسّطة. هذا ليس النموذج المدرَّب بعد، "
            "ولا يستخدم بيانات ناسا الحية."
        ),
        "in_soil": "رطوبة التربة (%)",
        "in_temp": "درجة حرارة الهواء (°م)",
        "in_ndvi": "مؤشر الغطاء النباتي (NDVI)",
        "result": "التوصية",
        "irrigate": "يُنصح بالري اليوم",
        "monitor": "راقب عن كثب",
        "none": "لا حاجة للري",
        "why": "العوامل المؤثرة",
        "r_soil_low": "رطوبة التربة منخفضة ({v:.0f}%).",
        "r_soil_mid": "رطوبة التربة متوسطة ({v:.0f}%).",
        "r_heat": "ارتفاع الحرارة يزيد الطلب على المياه ({v:.0f} °م).",
        "r_veg": "مؤشر الغطاء النباتي يدل على إجهاد المحصول (NDVI {v:.2f}).",
        "r_ok": "لا توجد مؤشرات إجهاد.",
        "in_rain": "الأمطار خلال آخر 7 أيام (مم)",
        "rain7": "الأمطار خلال آخر 7 أيام",
        "temp_air": "درجة حرارة الهواء",
        "ai_note": (
            "توقع من النموذج المدرَّب. تُدخل القيم يدوياً هنا ولا تُقرأ "
            "مباشرة من خدمات ناسا."
        ),
        "ai_missing": "النموذج المدرَّب غير متاح بعد، لذا تُعرض القواعد المبسّطة أدناه.",
        "ai_synthetic": (
            "اختبار للواجهة: دُرِّب هذا النموذج على بيانات اصطناعية. التوقع "
            "والتوصية والتفسير ليست نتائج حقيقية."
        ),
        "forecast": "توقع {label} بعد {h} أيام: {v:.1f} {unit}",
        "rec_caveat": (
            "العتبات توضيحية، ورطوبة التربة من SMAP إشارة إقليمية (~9 كم). "
            "تحقق مع المزارع قبل الري."
        ),
        "shap_t": "لماذا أعطى النموذج هذا التوقع (SHAP)",
        "shap_base": "متوسط توقع النموذج: {v:.1f}",
        "shap_up": "{f} = {x:.2f} رفع التوقع بمقدار {v:.2f}",
        "shap_down": "{f} = {x:.2f} خفّض التوقع بمقدار {v:.2f}",
        "shap_imputed": " (قيمة مفقودة عُوِّضت بقيمة نموذجية)",
        "shap_note": "توضح هذه القيم كيف يستخدم النموذج كل مدخل، وليست دليلاً على السببية.",
        "eval_t": "تقييم النموذج",
        "eval_line": (
            "خطأ الاختبار (MAE): النموذج {m:.2f}، والمرجع «بلا تغيير» {p:.2f} "
            "على {n} يوماً محجوزاً للاختبار."
        ),
        "env": "بيئات التشغيل",
        "tab_earth": "الأرض",
        "tab_mars": "المريخ (مفهوم)",
        "earth_text": "يدعم قرارات الري في المزارع العُمانية باستخدام بيانات مشتقة من الأقمار الصناعية.",
        "mars_text": (
            "امتداد مفاهيمي للزراعة في بيئات مغلقة يُعاد فيها تدوير المياه "
            "وتكون الموارد محدودة. يوضح كيف يمكن تكييف منطق القرار نفسه، "
            "ولا يستخدم بيانات زراعية حقيقية من المريخ."
        ),
        "m_param": "المعلمة",
        "m_value": "القيمة",
        "m_water": "مصدر المياه",
        "m_light": "الإضاءة",
        "m_temp": "التحكم بالحرارة",
        "m_crop": "احتياج المحصول من المياه",
        "m_pending": "قيد التحديد",
        "footer": (
            "طوّره فريق Red Oasis لتحدي ناسا لتطبيقات الفضاء 2026. "
            "هذا نموذج أولي مستقل وليس منتجاً رسمياً من ناسا. "
            "مصدر البيانات: ناسا."
        ),
    },
}

# ======================================================================
# PAGE SETUP AND STYLE
# ======================================================================

st.set_page_config(page_title="Red Oasis", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: 'IBM Plex Sans', 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif; }
.block-container { max-width: 1080px; padding-top: 1.5rem; }
header[data-testid="stHeader"] { background: transparent; }
.ro-hero { background: #0f2347; color: #ffffff; padding: 2.2rem 2rem; border-radius: 6px; margin: 0.5rem 0 1.6rem 0; }
.ro-hero h1 { color: #ffffff; font-size: 2.6rem; font-weight: 700; margin: 0 0 0.2rem 0; padding: 0; }
.ro-hero .tag { color: #e8b08a; font-size: 1.1rem; margin-bottom: 0.9rem; }
.ro-hero .sub { color: #d6def0; font-size: 1.05rem; max-width: 40rem; margin-bottom: 1rem; }
.ro-hero .badge { color: #b9c6e4; font-size: 0.9rem; border-top: 1px solid #2c4170; padding-top: 0.7rem; }
.ro-point { border-inline-start: 3px solid #b23a22; padding-inline-start: 0.9rem; margin-bottom: 0.6rem; }
.ro-point h4 { margin: 0 0 0.3rem 0; font-size: 1.05rem; color: #0f2347; }
.ro-point p { margin: 0; color: #3a455c; font-size: 0.95rem; }
.ro-facts { width: 100%; border-collapse: collapse; margin-bottom: 0.8rem; }
.ro-facts td { padding: 0.5rem 0.2rem; border-bottom: 1px solid #dfe4ee; font-size: 0.95rem; }
.ro-facts td:first-child { color: #5a667e; width: 38%; }
.ro-result { border-inline-start: 5px solid var(--c); background: #f4f6fb; padding: 0.9rem 1.1rem; border-radius: 4px; margin: 0.4rem 0 0.8rem 0; }
.ro-result .lvl { font-size: 1.35rem; font-weight: 700; color: var(--c); }
.ro-foot { color: #5a667e; font-size: 0.85rem; border-top: 1px solid #dfe4ee; padding-top: 0.9rem; margin-top: 2rem; }
h2, h3 { color: #0f2347; }
.stSlider { direction: ltr; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ======================================================================
# LANGUAGE
# ======================================================================

choice = st.radio(
    "Language / اللغة",
    ["English", "العربية"],
    horizontal=True,
    key="lang_choice",
)
lang = "en" if choice == "English" else "ar"
t = TEXT[lang]

if lang == "ar":
    st.markdown(
        "<style>.stApp { direction: rtl; } "
        ".stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp td, .stApp th "
        "{ text-align: right; }</style>",
        unsafe_allow_html=True,
    )

# ======================================================================
# HERO
# ======================================================================

st.markdown(
    f"""
    <div class="ro-hero">
        <h1>Red Oasis</h1>
        <div class="tag">{t['tagline']}</div>
        <div class="sub">{t['subtitle']}</div>
        <div class="badge">{t['badge']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# OVERVIEW
# ======================================================================

st.header(t["overview"])
c1, c2, c3 = st.columns(3)
for col, (title, body) in zip(
    (c1, c2, c3),
    (
        (t["problem_t"], t["problem"]),
        (t["approach_t"], t["approach"]),
        (t["explain_t"], t["explain"]),
    ),
):
    with col:
        st.markdown(
            f'<div class="ro-point"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

# ======================================================================
# TEST SITE
# ======================================================================

st.header(t["site"])
left, right = st.columns([2, 3])

crop = SITE["crop"][lang] if SITE["crop"] else t["crop_tbc"]
status = t["verified"] if SITE["verified"] else t["pending"]

with left:
    st.markdown(
        f"""
        <table class="ro-facts">
            <tr><td>{t['site_name']}</td><td>{SITE['name'][lang]}</td></tr>
            <tr><td>{t['site_region']}</td><td>{SITE['region'][lang]}</td></tr>
            <tr><td>{t['site_crop']}</td><td>{crop}</td></tr>
            <tr><td>{t['site_status']}</td><td>{status}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption(t["coords"].format(lat=SITE["lat"], lon=SITE["lon"]))
    if not SITE["verified"]:
        st.caption(t["site_note"])

with right:
    st.map(
        pd.DataFrame({"lat": [SITE["lat"]], "lon": [SITE["lon"]]}),
        latitude="lat",
        longitude="lon",
        zoom=11,
    )

# ======================================================================
# DATA SOURCES
# ======================================================================

st.header(t["data"])
rows = [
    {
        t["c_source"]: source,
        t["c_var"]: t[var_key],
        t["c_use"]: t[use_key],
        t["c_status"]: t["planned"],
    }
    for source, var_key, use_key in DATA_SOURCES
]
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(t["data_note"])

# ======================================================================
# IRRIGATION ADVISOR
# ======================================================================

st.header(t["advisor"])

i1, i2, i3, i4 = st.columns(4)
with i1:
    soil = st.slider(t["in_soil"], 0, 100, 25)
with i2:
    temp = st.slider(t["in_temp"], 10, 55, 41)
with i3:
    ndvi = st.slider(t["in_ndvi"], 0.0, 1.0, 0.25, step=0.01)
with i4:
    rain = st.slider(t["in_rain"], 0, 100, 0)

ai = get_ai_assessment(soil, temp, ndvi, rain)
use_ai = ai is not None and ai["status"] == "ok" and ai["recommendation"] is not None

if use_ai:
    st.caption(t["ai_note"])
    if ai["model"]["is_synthetic"]:
        st.warning(t["ai_synthetic"])
    level = ai["recommendation"]["level"]
else:
    st.caption(t["advisor_note"])
    st.info(t["ai_missing"])
    if ai is not None and ai["status"] in ("invalid_input", "error"):
        st.caption("; ".join(ai["errors"]))
    level, reasons = get_recommendation(soil, temp, ndvi)

colour = {"irrigate": "#b23a22", "monitor": "#b7791f", "none": "#2f7d4f"}[level]

st.markdown(
    f"""
    <div class="ro-result" style="--c:{colour}">
        <div>{t['result']}</div>
        <div class="lvl">{t[level]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if use_ai:
    pred = ai["prediction"]
    st.markdown(t["forecast"].format(
        label=t[FEATURE_TEXT[pred["target"]]],
        h=pred["horizon_days"],
        v=pred["value"],
        unit="%" if pred["target"] == "soil_moisture_pct" else "",
    ))
    st.caption(t["rec_caveat"])
    for warning in ai["warnings"]:
        if "SYNTHETIC" not in warning:
            st.caption(f"⚠ {warning}")

    exp = ai["explanation"]
    if exp:
        st.markdown(f"**{t['shap_t']}**")
        st.caption(t["shap_base"].format(v=exp["base_value"]))
        for c in exp["contributions"]:
            line = t["shap_up" if c["contribution"] >= 0 else "shap_down"].format(
                f=t[FEATURE_TEXT[c["feature"]]], x=c["value"], v=abs(c["contribution"])
            )
            st.markdown(f"- {line}{t['shap_imputed'] if c['imputed'] else ''}")
        st.bar_chart(
            pd.DataFrame({
                "factor": [t[FEATURE_TEXT[c["feature"]]] for c in exp["contributions"]],
                "SHAP": [c["contribution"] for c in exp["contributions"]],
            }),
            x="factor",
            y="SHAP",
            horizontal=True,
        )
        st.caption(t["shap_note"])

    evaluation = model_status()["evaluation"]
    if evaluation.get("status") == "ok":
        with st.expander(t["eval_t"]):
            metrics = evaluation["test_metrics"]
            st.write(t["eval_line"].format(
                m=metrics["model"]["mae"],
                p=metrics["baseline_persistence"]["mae"],
                n=evaluation["n_test_compared"],
            ))
            if evaluation.get("is_synthetic"):
                st.caption(evaluation["warning"])
else:
    st.markdown(f"**{t['why']}**")
    for key, value in reasons:
        st.markdown(f"- {t[key].format(v=value)}" if value is not None else f"- {t[key]}")

# ======================================================================
# OPERATING ENVIRONMENTS
# ======================================================================

st.header(t["env"])
tab_earth, tab_mars = st.tabs([t["tab_earth"], t["tab_mars"]])

with tab_earth:
    st.write(t["earth_text"])

with tab_mars:
    st.write(t["mars_text"])

    def mars_value(entry):
        return entry[lang] if entry else t["m_pending"]

    mars_rows = [
        {t["m_param"]: t["m_water"], t["m_value"]: mars_value(MARS_PARAMS["water"])},
        {t["m_param"]: t["m_light"], t["m_value"]: mars_value(MARS_PARAMS["light"])},
        {t["m_param"]: t["m_temp"], t["m_value"]: mars_value(MARS_PARAMS["temperature"])},
        {t["m_param"]: t["m_crop"], t["m_value"]: mars_value(MARS_PARAMS["crop_water"])},
    ]
    st.dataframe(pd.DataFrame(mars_rows), hide_index=True)

# ======================================================================
# FOOTER
# ======================================================================

st.markdown(f'<div class="ro-foot">{t["footer"]}</div>', unsafe_allow_html=True)