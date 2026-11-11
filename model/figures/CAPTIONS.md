# Figure captions (2 lines each, English and Arabic)

Owner: Yaqeen. Checked on 11 Nov 2026 against `model/figures/figures_data.json` and
`model/artifacts/evaluation.json` (final test: 122 days, 18 Mar to 27 Sep 2026). The model is
experimental: it beat "same as today" on the final test (1.20 vs 1.31) but not in
cross-validation (1.18 vs 1.03).

## fig1_forecast_vs_actual.png

EN:
Blue line: the forecast made 3 days earlier. Grey dots: what SMAP observed on the 122 test days.
The forecast is much flatter than reality (spread 1.1 vs 2.8 points) and misses sudden wetting after rain, as in late March.

AR:
الخط الأزرق: التوقع الصادر قبل 3 أيام. النقاط الرمادية: ما رصده SMAP فعلاً في أيام الاختبار الـ 122.
التوقع أكثر ثباتاً من الواقع بكثير (تشتت 1.1 مقابل 2.8 نقطة)، ولا يلتقط الترطّب المفاجئ بعد المطر كما في أواخر مارس.

## fig2_error_rain_vs_no_rain.png

EN:
Average error on the test days, with and without rain in the previous 7 days (lower is better).
The model is slightly better than "same as today" in both cases (1.57 vs 1.72 after rain, 0.63 vs 0.67 without), but errors after rain are more than twice as large.

AR:
متوسط الخطأ في أيام الاختبار، مع المطر وبدونه خلال الأيام السبعة السابقة (الأقل أفضل).
النموذج أفضل قليلاً من "مثل اليوم" في الحالتين (1.57 مقابل 1.72 بعد المطر، و0.63 مقابل 0.67 بدونه)، لكن الخطأ بعد المطر أكبر بأكثر من الضعف.

## fig3_shap_summary.png

EN:
Average effect of each input on the forecast (mean |SHAP|, percentage points): soil moisture today 0.40, recent rain 0.23, air temperature 0.21.
This shows how the model uses its inputs, not what physically changes the soil.

AR:
متوسط أثر كل مدخل على التوقع (متوسط |SHAP| بالنقاط المئوية): رطوبة التربة اليوم 0.40، المطر الأخير 0.23، حرارة الهواء 0.21.
هذا يصف كيف يستخدم النموذج مدخلاته، لا السبب الفيزيائي لتغيّر رطوبة التربة.
