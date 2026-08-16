import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================
# PAGE CONFIG & HEADER
# ============================================
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

col1, col2 = st.columns([4, 1])
with col1:
    st.title("❤️ Heart Disease – Model Comparison")
with col2:
    train_all_clicked = st.button("🚀 Train All Models", type="secondary", use_container_width=True)

if train_all_clicked:
    with st.spinner("⏳ Training all models... Please wait."):
        result = subprocess.run(
            [sys.executable, "train_all.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            st.success("✅ All models trained successfully!")
            with st.expander("📝 Training Output"):
                st.text(result.stdout)
            st.rerun()
        else:
            st.error("❌ Training failed. See details below:")
            st.text(result.stderr)

# ============================================
# SIDEBAR – MODEL SELECTION & TRAIN BUTTON
# ============================================
st.sidebar.header("⚙️ Settings")

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'KNN',
    'Naive Bayes',
    'Random Forest'
]

selected_model = st.sidebar.selectbox("Select Model", model_names)
model_file = f'model/{selected_model.lower().replace(" ", "_")}.pkl'

train_eval_clicked = st.sidebar.button("⚙️ Train & Evaluate Selected", type="primary", use_container_width=True)

if os.path.exists(model_file):
    st.sidebar.success(f"✅ {selected_model} is already trained.")
else:
    st.sidebar.warning(f"⚠️ {selected_model} not trained yet. Click the button to train it.")

# ============================================
# FILE UPLOAD / AUTO-LOAD test_data.csv
# ============================================
st.header("📂 Test Data")

# Check if test_data.csv exists in the current directory
default_file = 'test_data.csv'
uploaded_file = None

# Display file uploader; if default file exists, show it as preloaded
if os.path.exists(default_file):
    st.info(f"📄 Using default test data: `{default_file}` (auto‑generated)")
    # Read the default file
    default_data = pd.read_csv(default_file)
    st.success(f"✅ Default data loaded: {default_data.shape[0]} rows, {default_data.shape[1]} columns")
    with st.expander("🔍 Preview default data"):
        st.dataframe(default_data.head(10))
    # Offer the option to upload a different file
    uploaded_file = st.file_uploader("Or upload a different CSV file (optional)", type=['csv'])
else:
    st.warning("⚠️ No default `test_data.csv` found. Please upload a CSV file.")
    uploaded_file = st.file_uploader("Choose a CSV file (must have a 'target' column)", type=['csv'])

# Determine which data to use: uploaded file takes precedence over default
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded data loaded: {test_data.shape[0]} rows, {test_data.shape[1]} columns")
    with st.expander("🔍 Preview uploaded data"):
        st.dataframe(test_data.head(10))
elif os.path.exists(default_file):
    test_data = default_data
else:
    test_data = None

# ============================================
# EVALUATE IF DATA AND TRAIN BUTTON ARE READY
# ============================================
if test_data is not None:
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    if train_eval_clicked:
        # Check if model exists; if not, run its training script
        if not os.path.exists(model_file):
            script_map = {
                'Logistic Regression': 'model/logistic_regression_train.py',
                'Decision Tree': 'model/decision_tree_train.py',
                'KNN': 'model/knn_train.py',
                'Naive Bayes': 'model/naive_bayes_train.py',
                'Random Forest': 'model/random_forest_train.py'
            }
            script_path = script_map[selected_model]
            if not os.path.exists(script_path):
                st.error(f"❌ Training script '{script_path}' not found.")
                st.stop()

            with st.spinner(f"⏳ Training {selected_model}... Please wait."):
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    st.error(f"❌ Training failed for {selected_model}:")
                    st.text(result.stderr)
                    st.stop()
                else:
                    st.success(f"✅ {selected_model} trained successfully!")
                    if os.path.exists('test_data.csv'):
                        st.info("📁 `test_data.csv` has been created (or updated).")
                    st.rerun()
        else:
            st.info(f"ℹ️ {selected_model} is already trained. Loading it...")

        # Now load and evaluate the model
        try:
            # Load scaler
            try:
                scaler = joblib.load('model/scaler.pkl')
                X_test_scaled = scaler.transform(X_test)
                scaled = True
            except:
                X_test_scaled = X_test
                scaled = False
                st.warning("⚠️ Scaler not found – using raw features (may affect LR/KNN).")

            model = joblib.load(model_file)

            # Predict
            if selected_model in ['Logistic Regression', 'KNN']:
                if scaled:
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Display results
            st.subheader(f"📊 {selected_model} – Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC", f"{auc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col2.metric("Recall", f"{rec:.4f}")
            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease']
            )
            ax.set_title(f"{selected_model} – Confusion Matrix")
            st.pyplot(fig)

            # Classification Report
            st.subheader("📋 Detailed Report")
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                try:
                    st.dataframe(
                        report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                    )
                except AttributeError:
                    st.dataframe(report_df)
                    st.info("💡 Install `jinja2` for enhanced table styling: `pip install jinja2`")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

        except FileNotFoundError:
            st.error(f"❌ Model file '{model_file}' not found even after training. Please check the training script.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

    else:
        st.info("👈 Select a model and click **'Train & Evaluate Selected'** to get results.")

else:
    st.info("👆 Please upload a CSV file or generate `test_data.csv` by training a model.")
    st.markdown("""
    **Instructions:**
    1. Click **'Train All Models'** (top‑right) or train any individual model to generate `test_data.csv`.
    2. Alternatively, upload a CSV file where the **last column is the target** (0 = No Disease, 1 = Disease).
    3. Select a model from the sidebar.
    4. Click **'Train & Evaluate Selected'** – if the model is not trained yet, it will train it.
    5. View the results (metrics, confusion matrix, and classification report).
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("❤️ Built with Streamlit | UCI Heart Disease Dataset | ML Assignment 2")