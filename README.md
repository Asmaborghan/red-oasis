# Red Oasis

A NASA-data-informed irrigation advisor for date-palm farming in Oman, with an illustrative Mars water-management mode. Built for the NASA Space Apps Challenge 2026 (theme: The Next Frontier).

**Live app:** https://red-oasis.streamlit.app/

**Demo video (60 seconds):** [Watch on Google Drive](https://drive.google.com/file/d/1jPH7NvLFWnesFxPjYfnqHF8l89gn0lya/view?usp=drive_link)

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

## Data

NASA Earth observation data for the Dariz farm point (23.3086 N, 56.6131 E), 2024 to 2026: SMAP soil moisture, MODIS/VIIRS NDVI, GPM IMERG rainfall and NASA POWER weather. Published agricultural research is used for the site evidence and the Mars reference values.

## Model status

An experimental Random Forest forecasts soil moisture three days ahead. On held-out days its error is 1.20 percentage points, against 1.31 for the simple "no change" forecast, but it did not beat that baseline in cross-validation, so it is not yet a validated predictor. It does not predict crop stress. Details: `model/README.md`.

## Limitations

- SMAP soil moisture is a regional signal of about 9 km, not a measurement inside the orchard.
- The advisor thresholds are illustrative and must not be used for real irrigation decisions.
- The Mars mode is a concept based on NASA reference values and team assumptions, not a simulation.
- This is an independent student prototype, not an official NASA product. Data credit: NASA.

## Team

Asma (App Builder), Amna (NASA Data), Yaqeen (ML Model), Shareefa (Field Research), Jumana (Storytelling and Presentation).