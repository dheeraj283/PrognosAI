# =========================================================
# 📦 IMPORTS
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =========================================================
# ⚙️ PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PrognosAI Control Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 📂 LOAD DATASETS
# =========================================================
@st.cache_data
def load_all_datasets():
    datasets = {}
    for fd in ['fd001', 'fd002', 'fd003', 'fd004']:
        seq_data = np.load(f'{fd}_sequences.npz')
        eval_data = np.load(f'{fd}_evaluation.npz')

        y_true = eval_data['y_test']
        y_pred = eval_data['y_test_pred']

        datasets[fd] = {
            'X': seq_data['X'],
            'y_test': y_true,
            'y_test_pred': y_pred,
            'rmse': np.sqrt(np.mean((y_pred - y_true) ** 2)),
            'mae': np.mean(np.abs(y_pred - y_true)),
            'n_samples': len(y_true)
        }
    return datasets


# =========================================================
# 🧠 LOAD MODELS
# =========================================================
@st.cache_resource
def load_all_models():
    models = {}
    for fd in ['fd001', 'fd002', 'fd003', 'fd004']:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(30, 21)),
            tf.keras.layers.GRU(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.load_weights(f'{fd}_gru_model.weights.h5')
        models[fd] = model
    return models


datasets = load_all_datasets()
models = load_all_models()


# =========================================================
# 🚨 ALERT CLASSIFICATION
# =========================================================
def classify_alert(rul):
    if rul < 10:
        return "CRITICAL"
    elif rul < 30:
        return "WARNING"
    elif rul < 125:
        return "SAFE"
    else:
        return "VERY SAFE"


# =========================================================
# 🖥️ HEADER
# =========================================================
st.markdown("## 🧠 PrognosAI – Engine Health Control Center")
st.caption("GRU-based Remaining Useful Life Prediction | NASA CMAPSS")
st.markdown("---")


# =========================================================
# 🎛️ CONTROL PANEL
# =========================================================
c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    selected_fd = st.selectbox(
        "Dataset",
        list(datasets.keys()),
        format_func=lambda x: x.upper()
    )

with c2:
    view_mode = st.selectbox(
        "View Mode",
        ["📊 Dataset Summary", "🔍 Single Engine Analysis", "🧪 Prediction Explorer"]
    )

with c3:
    engine_ids = list(range(len(datasets[selected_fd]['X'])))
    selected_engine = st.selectbox("Engine ID", engine_ids)

st.markdown("---")


# =========================================================
# 📊 DATASET SUMMARY
# =========================================================
if view_mode == "📊 Dataset Summary":

    st.subheader(f"Dataset Overview — {selected_fd.upper()}")

    # ---- Metrics ----
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Samples", f"{datasets[selected_fd]['n_samples']:,}")
    m2.metric("RMSE", f"{datasets[selected_fd]['rmse']:.2f}")
    m3.metric("MAE", f"{datasets[selected_fd]['mae']:.2f}")

    st.markdown("---")

    # ---- Engine-wise table ----
    y_true = datasets[selected_fd]['y_test']
    y_pred = datasets[selected_fd]['y_test_pred']

    engine_table = pd.DataFrame({
        "Engine ID": np.arange(len(y_true)),
        "True RUL": y_true.astype(int),
        "Predicted RUL": y_pred.round(1),
        "Absolute Error": np.abs(y_true - y_pred).round(1),
        "Alert Status": [classify_alert(r) for r in y_pred]
    })

    st.markdown("### Engine-wise RUL & Alert Report")
    st.dataframe(engine_table, use_container_width=True)

    # ---- Alert statistics ----
    alerts = engine_table["Alert Status"]
    alert_counts = alerts.value_counts()
    alert_percent = (alert_counts / alert_counts.sum() * 100).round(1)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("🔴 CRITICAL", int(alert_counts.get("CRITICAL", 0)))
    a2.metric("🟡 WARNING", int(alert_counts.get("WARNING", 0)))
    a3.metric("🟢 SAFE", int(alert_counts.get("SAFE", 0)))
    a4.metric("🔵 VERY SAFE", int(alert_counts.get("VERY SAFE", 0)))

    # ---- Small graphs ----
    g1, g2 = st.columns(2)

    with g1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.bar(alert_counts.index, alert_counts.values)
        ax1.set_title("Engines per Alert")
        ax1.tick_params(axis='x', rotation=30)
        st.pyplot(fig1)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie(
            alert_percent.values,
            labels=alert_percent.index,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        ax2.set_title("Alert Distribution")
        st.pyplot(fig2)

    # ---- Downloads ----
    st.markdown("### 📥 Download Reports")

    st.download_button(
        "⬇ Download Engine Report (CSV)",
        engine_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_fd}_engine_rul_report.csv",
        mime="text/csv"
    )

# =========================================================
# 🔍 SINGLE ENGINE ANALYSIS (COMPACT + TABLE)
# =========================================================
elif view_mode == "🔍 Single Engine Analysis":

    st.subheader(f"Engine {selected_engine} — {selected_fd.upper()}")

    # ---- Data ----
    seq = datasets[selected_fd]['X'][selected_engine]
    predicted_rul = models[selected_fd].predict(
        seq[None, ...], verbose=0
    )[0, 0]
    alert_state = classify_alert(predicted_rul)

    # ---- Layout: half-size graph + table ----
    left, right = st.columns([1, 1])

    # ---- SMALL GRAPH ----
    with left:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(seq)
        ax.set_title("Sensor Signals (30 cycles)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        st.pyplot(fig)

    # ---- ENGINE DETAIL TABLE ----
    with right:
        engine_details = {
            "Dataset": selected_fd.upper(),
            "Engine ID": selected_engine,
            "Predicted RUL (cycles)": round(predicted_rul, 1),
            "Alert Status": alert_state,
            "Mean Sensor Value": round(np.mean(seq), 3),
            "Min Sensor Value": round(np.min(seq), 3),
            "Max Sensor Value": round(np.max(seq), 3),
            "Sensor Std Dev": round(np.std(seq), 3),
            "Dataset RMSE": round(datasets[selected_fd]["rmse"], 2),
            "Dataset MAE": round(datasets[selected_fd]["mae"], 2),
        }

        detail_df = pd.DataFrame(
            engine_details.items(),
            columns=["Metric", "Value"]
        )

        st.markdown("### Engine Diagnostics")
        st.table(detail_df)

# =========================================================
# 🧪 PREDICTION EXPLORER (COMPACT + ANALYTICS)
# =========================================================
elif view_mode == "🧪 Prediction Explorer":

    st.subheader(f"Prediction Explorer — {selected_fd.upper()}")

    actual = datasets[selected_fd]['y_test']
    predicted = datasets[selected_fd]['y_test_pred']
    errors = np.abs(actual - predicted)

    # ---- Layout: small graph + tables ----
    left, right = st.columns([1, 1])

    # ---- SMALL SCATTER PLOT ----
    with left:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax.scatter(actual, predicted, alpha=0.5, s=10)
        ax.plot([0, 200], [0, 200], 'k--', linewidth=1)
        ax.set_xlabel("Actual RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    # ---- ANALYTICS TABLES ----
    with right:
        st.markdown("### Prediction Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "Mean Error", "Max Error"],
            "Value": [
                round(datasets[selected_fd]["rmse"], 2),
                round(datasets[selected_fd]["mae"], 2),
                round(errors.mean(), 2),
                round(errors.max(), 2)
            ]
        })
        st.table(metrics_df)

        st.markdown("### Top 10 Highest Errors")

        error_table = pd.DataFrame({
            "Engine ID": np.arange(len(actual)),
            "True RUL": actual.astype(int),
            "Predicted RUL": predicted.round(1),
            "Absolute Error": errors.round(1),
            "Alert Status": [classify_alert(r) for r in predicted]
        })

        worst_10 = error_table.sort_values(
            by="Absolute Error", ascending=False
        ).head(10)

        st.dataframe(worst_10, use_container_width=True)

# =========================================================
# 🧾 FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "**PrognosAI Control Center**  \n"
    "GRU-based Prognostics | NASA CMAPSS | Compact Analytics Dashboard"
)
