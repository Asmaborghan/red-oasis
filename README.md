# Red Oasis 🌱

> **A Smart Agriculture Advisor Powered by NASA Data — From the Omani Desert to Mars.**

---

## 📌 Overview
**Red Oasis** is a web-based decision support system designed for extreme agriculture. By utilizing open-source Earth observation data from NASA, Red Oasis predicts crop water stress, optimizes irrigation scheduling, and explains recommendations to farmers using explainable AI (XAI). 

Additionally, Red Oasis features a conceptual **Mars Mode** to assist astronauts in managing closed-loop agricultural environments in space.

---

## 🎯 Key Features
* 📡 **NASA Earth Data Integration:** Utilizes remote sensing data (SMAP soil moisture, MODIS/VIIRS NDVI, GPM precipitation, ECOSTRESS water stress, and NASA POWER climate metrics).
* 🤖 **Predictive AI:** Machine learning models forecasting crop water stress days in advance.
* 💡 **Explainable AI (XAI):** Built-in SHAP explanations providing transparent rationale behind every irrigation advice.
* 🚀 **Dual-Environment (Earth & Space):** Dedicated mode tailored for simulated closed-system farming on the Moon and Mars.
* 🌍 **Local Relevance:** Optimized for arid climate conditions in Oman and desert regions.

---

## 🛠️ Project Structure
```text
red-oasis/
├── app/        # Streamlit user interface & web app code
├── data/       # Data ingestion, spatial inputs, & preprocessing scripts
├── model/      # Machine learning models, training scripts, & SHAP explainers
└── docs/       # Documentation, project slides, and research assets
