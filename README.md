# PrognosAI – Remaining Useful Life Prediction

PrognosAI is an end-to-end **predictive maintenance** project that predicts the **Remaining Useful Life (RUL)** of aircraft engines using **GRU-based deep learning** on multivariate time-series sensor data.

---

## 🔍 Overview

The goal of this project is to estimate how many operational cycles an engine has left before failure. It uses historical sensor data to learn degradation patterns and converts predictions into **actionable alert levels** for maintenance decisions.

---

## 📊 Dataset

* **NASA CMAPSS** benchmark dataset
* Multivariate time-series data
* 21 sensor readings per cycle
* Multiple operating conditions (FD001–FD004)

---

## 🧠 Model

* **Architecture:** GRU-based neural network
* **Input:** Sliding windows of 30 cycles × 21 sensors
* **Output:** Predicted Remaining Useful Life (RUL)
* **Why GRU:** Captures temporal dependencies with lower complexity than LSTM

**Evaluation Metrics:** RMSE, MAE

---

## 🛠️ Project Structure

```
├── Prognos_Time_Series.ipynb   # Data preprocessing & model training
├── prognos_dashboard.py        # Streamlit dashboard (deployment)
├── *_gru_model.weights.h5      # Trained model weights
├── *_sequences.npz             # Input sequences
├── *_evaluation.npz            # True & predicted RUL
```

---

## 📈 Dashboard Features

* Dataset-level performance summary (RMSE, MAE)
* Engine-wise RUL prediction and alert status
* Single engine sensor analysis
* Actual vs predicted RUL comparison
* CSV report download

**Alert Levels:** VERY SAFE, SAFE, WARNING, CRITICAL

---

## ▶️ Run the Project

```bash
pip install streamlit numpy pandas matplotlib tensorflow
streamlit run prognos_dashboard.py
```

---

## 👤 Author

**Ganti Dheeraj**
B.Tech ECE | Interests: Signal Processing, Machine Learning, AI
