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

import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st

# The AI component lives in model/ at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from model.predict import model_status, predict
except Exception:  # keep the website working if ML packages are missing
    predict = model_status = None
try:
    from model.irrigation import advise
except Exception:  # the rule is optional: the website keeps working without it
    advise = None

# ======================================================================
# CONFIGURATION
# ======================================================================

SITE = {
    "name": {"en": "Ad Dariz agricultural area", "ar": "منطقة الدريز الزراعية"},
    "region": {"en": "Ibri, Ad Dhahirah", "ar": "عبري، الظاهرة"},
    "lat": 23.308639,
    "lon": 56.613083,
    "verified": True,
    "crop": {"en": "Date palm", "ar": "نخيل التمر"},
    "area_ha": None,  # not established: the 255.44 ha in the research brief is the whole historic falaj, not this point
}

# Illustrative thresholds for the demonstration advisor.
# Soil moisture cut-offs are the 25th and 75th percentile of the Dariz SMAP record
# (7.0-27.4 %, median 8.8 %), so they describe "dry/normal for this site", not a
# crop requirement. TODO (Shareefa): replace with crop-specific values when available.
THRESHOLDS = {
    "soil_dry": 8.4,        # forecast soil moisture (%) below this: irrigate
    "soil_moderate": 9.3,   # below this: monitor
    "temp_high": 40,     # air temperature (deg C)
    "ndvi_low": 0.30,    # vegetation index below which stress is flagged
}

# Published values from Shareefa's research brief.
# Dariz data are historical; the irrigation trial is from a different location.
CROP_RESEARCH = {
    "dariz_study_period": "2002–2003",
    "crop": "Date palm (Phoenix dactylifera)",
    "annual_demand_mm": 3620,
    "annual_supply_mm_including_effective_rain": 3235,
    "demand_supply_ratio": 1.12,
    "shortfall_mm": 385,
    "demand_m3_ha_year": 36200,
    "supply_m3_ha_year": 32350,
    "fruit_stages_after_pollination": ["Hababouk", "Kimri", "Khalal", "Rutab", "Tamar"],
    "oman_trial_period": "2015–2016",
    "oman_trial_location": "Al Kamil and Al Wafi Agricultural Research Station",
    "oman_trial_variety": "Khalas",
    "bubbler_water_m3_ha_year": 9966,
    "subsurface_drip_60pct_etc_water_m3_ha_year": 6421,
    "bubbler_yield_kg_per_palm": 79.0,
    "subsurface_drip_60pct_etc_yield_kg_per_palm": 78.3,
    "subsurface_drip_40pct_etc_yield_kg_per_palm": 69.8,
    "subsurface_drip_20pct_etc_yield_kg_per_palm": 68.8,
}


# NASA references are for closed-environment concepts and ISS lettuce.
# They are not validated Mars conditions or date-palm growing requirements.
MARS_PARAMS = {
    "water": {
        "en": (
            "Closed-loop water recovery ON. Team stress tests: external "
            "make-up water at 50%, 75%, and 100% of the Earth baseline. "
            "These percentages are team assumptions, not NASA Mars values."
        ),
        "ar": (
            "إعادة تدوير المياه في دورة مغلقة. اختبارات الفريق: ماء تعويض "
            "خارجي بنسبة 50٪ و75٪ و100٪ من مرجع الأرض. هذه افتراضات للفريق "
            "وليست قيماً منشورة من ناسا للمريخ."
        ),
    },
    "light": {
        "en": (
            "Artificial; NASA ISS lettuce reference: "
            "16 h light / 8 h dark, 200 µmol m⁻² s⁻¹."
        ),
        "ar": (
            "إضاءة صناعية؛ مرجع ناسا لخس محطة الفضاء: "
            "16 ساعة إضاءة / 8 ساعات ظلام، 200 ميكرومول م⁻² ث⁻¹."
        ),
    },
    "temperature": {
        "en": (
            "NASA plant-growth reference: 10–35°C and 40–85% RH. "
            "ISS lettuce reference: 23.25°C."
        ),
        "ar": (
            "مرجع ناسا العام لنمو النباتات: 10–35°م ورطوبة نسبية 40–85٪. "
            "مرجع خس محطة الفضاء: 23.25°م."
        ),
    },
    "crop_water": {
        "en": (
            "No validated Mars water requirement for date palm. "
            "Dariz Earth reference (2002–2003): demand 3,620 mm/year; "
            "supply 3,235 mm/year."
        ),
        "ar": (
            "لا توجد قيمة موثقة لاحتياج نخيل التمر من الماء على المريخ. "
            "مرجع فلج الدريز على الأرض (2002–2003): الطلب 3,620 مم/سنة، "
            "والإمداد 3,235 مم/سنة."
        ),
    },
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


def get_ai_assessment(soil: float, temp: float, ndvi: float, rain: float, et0: float):
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
            "et0_mm": et0,
        },
        thresholds=THRESHOLDS,
    )


def get_rule_assessment(soil: float, temp: float, ndvi: float, rain: float, et0: float):
    """Call Yaqeen's provisional irrigation rule (model/irrigation.py).
    Returns its result dict, or None if it is unavailable or not usable."""
    if advise is None:
        return None
    try:
        out = advise(
            {
                "soil_moisture_pct": soil,
                "ndvi": ndvi,
                "rainfall_7d_mm": rain,
                "temperature_c": temp,
                "et0_mm": et0,
            }
        )
    except Exception:
        return None
    if out.get("status") == "ok" and out.get("level") in ("irrigate", "monitor", "none"):
        return out
    return None


# Model feature name -> TEXT key for its display label.
FEATURE_TEXT = {
    "soil_moisture_pct": "soil",
    "ndvi": "ndvi",
    "rainfall_7d_mm": "rain7",
    "temperature_c": "temp_air",
    "et0_mm": "et0",
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
            "rainfall and weather to forecast soil moisture and advise "
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
        "verified": "Satellite-reviewed candidate point",
        "coords": "Latitude {lat:.5f}, longitude {lon:.5f}",
        "site_note": "Coordinates are provisional until confirmed on satellite imagery.",
        "data": "NASA data sources",
        "c_source": "Source",
        "c_var": "Variable",
        "c_use": "Use in the model",
        "c_status": "Status",
        "planned": "Planned",
        "used": "Used (Dariz, 2024-2026)",
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
        "in_et0": "Reference evapotranspiration ET0 (mm/day)",
        "rain7": "Rainfall, last 7 days",
        "temp_air": "Air temperature (daily mean)",
        "et0": "Reference evapotranspiration (ET0)",
        "ai_note": (
            "Forecast from the trained model. Inputs are entered manually here, "
            "not read live from NASA services. Air temperature is treated as the daily mean."
        ),
        "ai_location_ok": (
            "The model was trained on NASA data for the Ad Dariz study point "
            "({lat:.2f}° N, {lon:.2f}° E). SMAP soil moisture is a regional (~9 km) "
            "signal, and the model is experimental."
        ),
        "ai_location": (
            "The model was trained on NASA data for a test point near Nizwa "
            "({site}, {lat:.2f}° N, {lon:.2f}° E), not for the Dariz farm shown above. "
            "Data for the exact farm location are still pending."
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
        "eval_t": "Model evaluation (real NASA-derived data)",
        "eval_line": (
            "Average forecast error (MAE) on {n} held-out days ({start} to {end}): "
            "model {m:.2f}, 'no change' baseline {p:.2f}, 'always the average' baseline {a:.2f} "
            "percentage points of soil moisture."
        ),
        "eval_cv_worse": (
            "Within the earlier training period the model did not beat the "
            "'no change' baseline, so its advantage is not yet reliable."
        ),
        "eval_mean_close": (
            "Soil moisture here changes little except after rain, so simple "
            "baselines are hard to beat."
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
        "rule_t": "How the level was decided (provisional rule)",
        "rule_caveat": (
            "The level comes from a provisional rule that compares today's soil "
            "moisture, the forecast, recent rain and ET0 with this site's own "
            "satellite record. It is not validated irrigation advice. Confirm "
            "with the grower before acting."
        ),
        "chk_soil_now_dry": "Soil moisture now {v:.1f}% vs site median {r:.1f}%",
        "chk_forecast_dry": "Forecast in 3 days {v:.1f}% vs site median {r:.1f}%",
        "chk_rain_below_demand": "Rain in the last 7 days {v:.1f} mm vs 3-day water demand {r:.1f} mm (3 × ET0)",
        "chk_high_demand": "ET0 today {v:.1f} mm/day vs site median {r:.1f} mm/day",
        "chk_yes": "points to irrigation",
        "chk_no": "does not",
        "chk_skipped": "Check skipped: an input is missing.",
        "limits": "Project limits",
        "lim1": "Regional, not field-level. One SMAP value covers about 9 km; it describes the surrounding surface soil, not the root zone of irrigated palms, and it cannot see falaj water inside the orchard.",
        "lim2": "The model forecasts soil moisture three days ahead. It does not measure crop stress, irrigation need or yield.",
        "lim3": "The forecast is nearly flat and misses sudden rain events. The model is experimental: in cross-validation it did not beat the simple 'same as today' forecast.",
        "lim4": "Short record: under three years for one location (2024-2026). A longer record is being prepared.",
        "lim5": "The irrigation levels come from a provisional rule relative to this site's satellite record. No field measurements or grower logs exist to validate it.",
        "lim6": "SHAP shows how the model uses each input, not what physically changes the soil.",
        "lim7": "The Mars mode is a concept built on NASA reference values and team assumptions, not a simulation.",
        "evid": "Site evidence",
        "evid_note": (
            "Published study of Falaj al-Dariz (Al-Ghafri, 2006; observations "
            "2002-2003). These are historical values, not current measurements."
        ),
        "evid_demand": "Crop water demand (date palm)",
        "evid_supply": "Water supplied (including effective rain)",
        "evid_unit": "Millimetres per year",
        "evid_demand_short": "Demand",
        "evid_supply_short": "Supply",
        "evid_short": "Shortfall (mm per year)",
        "evid_ratio": "Demand / supply ratio",
        "evid_caption": "Demand exceeded supply by {s:,} mm a year, about {p:.1f}% of demand.",
        "evid_trial_t": "Published comparison from another location",
        "evid_trial": (
            "At the Al Kamil and Al Wafi research station (Khalas palms, 2015-2016), "
            "subsurface drip at 60% of crop water demand used {w2:,} m³/ha a year "
            "versus {w1:,} m³/ha with bubbler irrigation ({pct:.1f}% less), with a "
            "similar yield ({y2} vs {y1} kg per palm). This is a published result "
            "from a different site, not a Red Oasis measurement."
        ),
        "mars_whatif": "Water what-if (illustrative)",
        "mars_baseline": "Earth baseline: Dariz date-palm demand of {d:,} m³/ha per year.",
        "mars_makeup": "External make-up water (% of Earth baseline)",
        "mars_recovery": "Water recovered and reused (%)",
        "mars_supplied": "Water available to the crop (m³/ha per year)",
        "mars_met": "Share of crop demand met",
        "mars_gap": "Shortfall (m³/ha per year)",
        "mars_ok": "Demand is met in this scenario.",
        "mars_short": "The crop would be short of water in this scenario.",
        "mars_assump": (
            "Illustrative arithmetic: water available = make-up water ÷ (1 − recovery). "
            "The recovery rate and the 50/75/100% levels are team assumptions, not "
            "NASA values, and this is not a simulation of a Mars greenhouse."
        ),
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
            "والأمطار والطقس لتوقّع رطوبة التربة والتوصية بموعد الري."
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
        "verified": "نقطة مرشحة روجعت على صور الأقمار الصناعية",
        "coords": "خط العرض {lat:.5f}، خط الطول {lon:.5f}",
        "site_note": "الإحداثيات مؤقتة إلى أن تُؤكَّد على صور الأقمار الصناعية.",
        "data": "مصادر بيانات ناسا",
        "c_source": "المصدر",
        "c_var": "المتغير",
        "c_use": "الاستخدام في النموذج",
        "c_status": "الحالة",
        "planned": "مخطط",
        "used": "مستخدم (الدريز، 2024-2026)",
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
        "in_et0": "التبخر-نتح المرجعي ET0 (مم/يوم)",
        "rain7": "الأمطار خلال آخر 7 أيام",
        "temp_air": "درجة حرارة الهواء (المتوسط اليومي)",
        "et0": "التبخر-نتح المرجعي (ET0)",
        "ai_note": (
            "توقع من النموذج المدرَّب. تُدخل القيم يدوياً هنا ولا تُقرأ مباشرة "
            "من خدمات ناسا. تُعامل درجة الحرارة على أنها المتوسط اليومي."
        ),
        "ai_location_ok": (
            "دُرِّب النموذج على بيانات ناسا لنقطة الدراسة في منطقة الدريز "
            "({lat:.2f}° شمالاً، {lon:.2f}° شرقاً). رطوبة التربة من SMAP إشارة "
            "إقليمية (نحو 9 كم)، والنموذج تجريبي."
        ),
        "ai_location": (
            "دُرِّب النموذج على بيانات ناسا لنقطة اختبار قرب نزوى "
            "({site}، {lat:.2f}° شمالاً، {lon:.2f}° شرقاً)، وليس لمزرعة الدريز المعروضة أعلاه. "
            "بيانات موقع المزرعة الدقيق ما زالت قيد الانتظار."
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
        "eval_t": "تقييم النموذج (بيانات حقيقية مشتقة من ناسا)",
        "eval_line": (
            "متوسط خطأ التوقع (MAE) على {n} يوماً محجوزاً للاختبار ({start} إلى {end}): "
            "النموذج {m:.2f}، مرجع «بلا تغيير» {p:.2f}، مرجع «المتوسط دائماً» {a:.2f} "
            "نقطة مئوية من رطوبة التربة."
        ),
        "eval_cv_worse": (
            "خلال فترة التدريب السابقة لم يتفوق النموذج على مرجع «بلا تغيير»، "
            "لذا فإن أفضليته ليست موثوقة بعد."
        ),
        "eval_mean_close": (
            "رطوبة التربة هنا قليلة التغير إلا بعد المطر، لذا يصعب التفوق على المراجع البسيطة."
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
        "rule_t": "كيف تحدد المستوى (قاعدة مؤقتة)",
        "rule_caveat": (
            "يأتي المستوى من قاعدة مؤقتة تقارن رطوبة التربة اليوم والتوقع والمطر "
            "الأخير والتبخر-نتح بسجل الأقمار الصناعية لهذا الموقع نفسه. وهي ليست "
            "نصيحة ري موثقة. تأكد من المزارع قبل التنفيذ."
        ),
        "chk_soil_now_dry": "رطوبة التربة الآن {v:.1f}% مقابل وسيط الموقع {r:.1f}%",
        "chk_forecast_dry": "التوقع بعد 3 أيام {v:.1f}% مقابل وسيط الموقع {r:.1f}%",
        "chk_rain_below_demand": "أمطار آخر 7 أيام {v:.1f} ملم مقابل احتياج 3 أيام {r:.1f} ملم (3 × ET0)",
        "chk_high_demand": "التبخر-نتح اليوم {v:.1f} ملم/يوم مقابل وسيط الموقع {r:.1f} ملم/يوم",
        "chk_yes": "يشير إلى الري",
        "chk_no": "لا يشير إلى الري",
        "chk_skipped": "تم تخطي الفحص: مدخل ناقص.",
        "limits": "حدود المشروع",
        "lim1": "إقليمي وليس على مستوى الحقل. قيمة SMAP الواحدة تغطي نحو 9 كم، وتصف التربة السطحية المحيطة لا منطقة جذور النخيل المروي، ولا ترى مياه الفلج داخل البستان.",
        "lim2": "النموذج يتوقع رطوبة التربة بعد ثلاثة أيام. ولا يقيس إجهاد المحصول أو الحاجة للري أو الإنتاج.",
        "lim3": "التوقع شبه ثابت ويفوّت أحداث المطر المفاجئة. والنموذج تجريبي: لم يتفوق في التحقق المتقاطع على توقع «مثل اليوم» البسيط.",
        "lim4": "سجل قصير: أقل من ثلاث سنوات لموقع واحد (2024-2026). ويجري تجهيز سجل أطول.",
        "lim5": "مستويات الري تأتي من قاعدة مؤقتة نسبةً إلى سجل الأقمار الصناعية لهذا الموقع. ولا توجد قياسات حقلية أو سجلات مزارعين للتحقق منها.",
        "lim6": "يوضح SHAP كيف يستخدم النموذج كل مدخل، لا ما يغيّر التربة فيزيائياً.",
        "lim7": "وضع المريخ مفهوم مبني على قيم مرجعية من ناسا وافتراضات الفريق، وليس محاكاة.",
        "evid": "أدلة الموقع",
        "evid_note": (
            "دراسة منشورة عن فلج الدريز (الغافري، 2006؛ ملاحظات 2002-2003). "
            "هذه قيم تاريخية وليست قياسات حالية."
        ),
        "evid_demand": "احتياج المحصول من الماء (نخيل التمر)",
        "evid_supply": "الماء المتوفر (شاملاً المطر الفعّال)",
        "evid_unit": "ملمتر في السنة",
        "evid_demand_short": "الطلب",
        "evid_supply_short": "الإمداد",
        "evid_short": "العجز (ملم في السنة)",
        "evid_ratio": "نسبة الطلب إلى الإمداد",
        "evid_caption": "تجاوز الطلب الإمداد بمقدار {s:,} ملم في السنة، أي نحو {p:.1f}% من الطلب.",
        "evid_trial_t": "مقارنة منشورة من موقع آخر",
        "evid_trial": (
            "في محطة الكامل والوافي للأبحاث (نخيل الخلاص، 2015-2016)، استهلك الري "
            "بالتنقيط تحت السطحي عند 60% من احتياج المحصول {w2:,} م³/هكتار في السنة "
            "مقابل {w1:,} م³/هكتار بالري بالفقاعات (أقل بنسبة {pct:.1f}%)، مع إنتاج "
            "مماثل ({y2} مقابل {y1} كجم للنخلة). هذه نتيجة منشورة من موقع مختلف، "
            "وليست قياساً من Red Oasis."
        ),
        "mars_whatif": "سيناريو الماء (توضيحي)",
        "mars_baseline": "مرجع الأرض: احتياج نخيل الدريز {d:,} م³/هكتار في السنة.",
        "mars_makeup": "ماء التعويض الخارجي (% من مرجع الأرض)",
        "mars_recovery": "الماء المُسترجَع والمُعاد استخدامه (%)",
        "mars_supplied": "الماء المتاح للمحصول (م³/هكتار في السنة)",
        "mars_met": "نسبة تغطية احتياج المحصول",
        "mars_gap": "العجز (م³/هكتار في السنة)",
        "mars_ok": "الاحتياج مغطّى في هذا السيناريو.",
        "mars_short": "سيعاني المحصول من نقص الماء في هذا السيناريو.",
        "mars_assump": (
            "حساب توضيحي: الماء المتاح = ماء التعويض ÷ (1 − نسبة الاسترجاع). "
            "نسبة الاسترجاع ومستويات 50/75/100% افتراضات من الفريق وليست قيماً من ناسا، "
            "وهذا ليس محاكاة لبيت زراعي على المريخ."
        ),
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

# Team logo (optional: the page still works if the file is missing).
LOGO_PATH = Path(__file__).parent / "assets" / "red_oasis_logo_v3_patch.png"


def load_logo():
    """Return (PIL image for the browser tab, base64 PNG for the banner) or (None, '')."""
    try:
        from PIL import Image

        image = Image.open(LOGO_PATH).convert("RGBA")
        small = image.resize((192, 192))
        buffer = io.BytesIO()
        small.save(buffer, format="PNG")
        return image.resize((64, 64)), base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None, ""


LOGO_ICON, LOGO_B64 = load_logo()

st.set_page_config(page_title="Red Oasis", page_icon=LOGO_ICON, layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: 'IBM Plex Sans', 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif; }
.stApp [data-testid="stIconMaterial"], .stApp span[class*="material-symbols"], .stApp span[class*="material-icons"] { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important; }
.block-container { max-width: 1080px; padding-top: 1.5rem; }
header[data-testid="stHeader"] { background: transparent; }
.ro-hero { background: #0f2347; color: #ffffff; padding: 2.2rem 2rem; border-radius: 6px; margin: 0.5rem 0 1.6rem 0; display: flex; align-items: center; gap: 1.6rem; }
.ro-hero img { width: 112px; height: 112px; flex-shrink: 0; }
@media (max-width: 640px) { .ro-hero { flex-direction: column; align-items: flex-start; } .ro-hero img { width: 84px; height: 84px; } }
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

logo_html = (
    f'<img src="data:image/png;base64,{LOGO_B64}" alt="Red Oasis logo">' if LOGO_B64 else ""
)

st.markdown(
    f"""
    <div class="ro-hero">
        {logo_html}
        <div>
            <h1>Red Oasis</h1>
            <div class="tag">{t['tagline']}</div>
            <div class="sub">{t['subtitle']}</div>
            <div class="badge">{t['badge']}</div>
        </div>
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
# SITE EVIDENCE (published values, see CROP_RESEARCH)
# ======================================================================

st.header(t["evid"])
st.caption(t["evid_note"])
ev_left, ev_right = st.columns([3, 2])
with ev_left:
    st.bar_chart(
        pd.DataFrame(
            {t["evid_unit"]: [
                CROP_RESEARCH["annual_demand_mm"],
                CROP_RESEARCH["annual_supply_mm_including_effective_rain"],
            ]},
            index=[t["evid_demand_short"], t["evid_supply_short"]],
        )
    )
with ev_right:
    st.metric(t["evid_short"], f"{CROP_RESEARCH['shortfall_mm']:,}")
    st.metric(t["evid_ratio"], f"{CROP_RESEARCH['demand_supply_ratio']:.2f}")
st.write(
    t["evid_caption"].format(
        s=CROP_RESEARCH["shortfall_mm"],
        p=100 * CROP_RESEARCH["shortfall_mm"] / CROP_RESEARCH["annual_demand_mm"],
    )
)
_w1 = CROP_RESEARCH["bubbler_water_m3_ha_year"]
_w2 = CROP_RESEARCH["subsurface_drip_60pct_etc_water_m3_ha_year"]
with st.expander(t["evid_trial_t"]):
    st.write(
        t["evid_trial"].format(
            w1=_w1,
            w2=_w2,
            pct=100 * (_w1 - _w2) / _w1,
            y1=CROP_RESEARCH["bubbler_yield_kg_per_palm"],
            y2=CROP_RESEARCH["subsurface_drip_60pct_etc_yield_kg_per_palm"],
        )
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
        t["c_status"]: t["used"],
    }
    for source, var_key, use_key in DATA_SOURCES
]
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(t["data_note"])

# ======================================================================
# IRRIGATION ADVISOR
# ======================================================================

st.header(t["advisor"])

i1, i2, i3 = st.columns(3)
with i1:
    soil = st.slider(t["in_soil"], 5.0, 35.0, 8.8, step=0.1)
with i2:
    temp = st.slider(t["in_temp"], 10, 50, 32)
with i3:
    ndvi = st.slider(t["in_ndvi"], 0.0, 1.0, 0.42, step=0.01)
i4, i5, _ = st.columns(3)
with i4:
    rain = st.slider(t["in_rain"], 0, 100, 0)
with i5:
    et0 = st.slider(t["in_et0"], 0.0, 15.0, 6.0, step=0.1)

ai = get_ai_assessment(soil, temp, ndvi, rain, et0)
use_ai = ai is not None and ai["status"] == "ok" and ai["recommendation"] is not None
rule = get_rule_assessment(soil, temp, ndvi, rain, et0) if use_ai else None

if use_ai:
    st.caption(t["ai_note"])
    if ai["model"]["is_synthetic"]:
        st.warning(t["ai_synthetic"])
    location = ai["model"].get("data_location") or {}
    if location.get("lat") and location.get("lon"):
        same_site = (
            abs(location["lat"][0] - SITE["lat"]) < 0.1
            and abs(location["lon"][0] - SITE["lon"]) < 0.1
        )
        st.info(t["ai_location_ok" if same_site else "ai_location"].format(
            site=", ".join(location.get("sites", ["?"])),
            lat=location["lat"][0],
            lon=location["lon"][0],
        ))
    level = rule["level"] if rule else ai["recommendation"]["level"]
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
        label=t[FEATURE_TEXT[pred["target"]]].lower() if lang == "en" else t[FEATURE_TEXT[pred["target"]]],
        h=pred["horizon_days"],
        v=pred["value"],
        unit="%" if pred["target"] == "soil_moisture_pct" else "",
    ))
    st.caption(t["rule_caveat"] if rule else t["rec_caveat"])
    for warning in ai["warnings"]:
        if "SYNTHETIC" not in warning:
            st.caption(f"⚠ {warning}")

    if rule:
        st.markdown(f"**{t['rule_t']}**")
        for check in rule["rule"]["checks"]:
            if check["met"] is None:
                st.markdown(f"- {t['chk_skipped']}")
            else:
                text = t["chk_" + check["name"]].format(v=check["value"], r=check["reference"])
                st.markdown(f"- {text} ({t['chk_yes'] if check['met'] else t['chk_no']})")

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

    # Metrics are shown only for a model trained on real data.
    evaluation = model_status()["evaluation"]
    if evaluation.get("status") == "ok" and not evaluation.get("is_synthetic"):
        try:
            metrics = evaluation["test_metrics"]
            eval_text = t["eval_line"].format(
                n=evaluation["n_test"],
                start=evaluation["test_period"][0],
                end=evaluation["test_period"][1],
                m=metrics["random_forest"]["mae"],
                p=metrics["baseline_persistence"]["mae"],
                a=metrics["baseline_training_mean"]["mae"],
            )
        except (KeyError, TypeError, IndexError):
            eval_text = None  # evaluation file has an unexpected layout: show nothing
        if eval_text:
            with st.expander(t["eval_t"]):
                st.write(eval_text)
                if evaluation.get("model_beats_persistence_in_cv") is False:
                    st.caption(t["eval_cv_worse"])
                st.caption(t["eval_mean_close"])
else:
    st.markdown(f"**{t['why']}**")
    for key, value in reasons:
        st.markdown(f"- {t[key].format(v=value)}" if value is not None else f"- {t[key]}")

# ======================================================================
# OPERATING ENVIRONMENTS
# ======================================================================
# PROJECT LIMITS
# ======================================================================

st.header(t["limits"])
for _key in ("lim1", "lim2", "lim3", "lim4", "lim5", "lim6", "lim7"):
    st.markdown(f"- {t[_key]}")

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
    st.table(pd.DataFrame(mars_rows).set_index(t["m_param"]))

    st.subheader(t["mars_whatif"])
    _demand = CROP_RESEARCH["demand_m3_ha_year"]
    st.caption(t["mars_baseline"].format(d=_demand))
    w1_, w2_ = st.columns(2)
    with w1_:
        makeup = st.select_slider(t["mars_makeup"], options=[50, 75, 100], value=75)
    with w2_:
        recovery = st.slider(t["mars_recovery"], 0, 95, 0, step=5)
    available = (makeup / 100) * _demand / (1 - recovery / 100)
    met = min(100.0, 100 * available / _demand)
    gap = max(0.0, _demand - available)
    m1, m2, m3 = st.columns(3)
    m1.metric(t["mars_supplied"], f"{available:,.0f}")
    m2.metric(t["mars_met"], f"{met:.0f}%")
    m3.metric(t["mars_gap"], f"{gap:,.0f}")
    if gap > 0:
        st.warning(t["mars_short"])
    else:
        st.success(t["mars_ok"])
    st.caption(t["mars_assump"])

# ======================================================================
# FOOTER
# ======================================================================

st.markdown(f'<div class="ro-foot">{t["footer"]}</div>', unsafe_allow_html=True)