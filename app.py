# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("❤️ Heart Disease – Model Comparison")

# Sidebar – Model selection
st.sidebar.header("⚙️ Choose Model")
model_options = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}
selected_model_name = st.sidebar.selectbox("Select a model", list(model_options.keys()))
model_file = model_options[selected_model_name]

# File upload
uploaded_file = st.file_uploader("📂 Upload test data (CSV)", type=['csv'])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success(f"✅ Data loaded: {test_data.shape[0]} rows")

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Load scaler and model
    try:
        scaler = joblib.load('model/scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
    except:
        X_test_scaled = X_test
        st.warning("⚠️ Scaler not found – using raw features.")

    try:
        model = joblib.load(f'model/{model_file}')
        # Predict
        if selected_model_name in ['Logistic Regression', 'KNN']:
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
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

        # Display metrics
        st.subheader(f"📊 {selected_model_name} – Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("AUC", f"{auc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col2.metric("Recall", f"{rec:.4f}")
        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_title(f"{selected_model_name}")
        st.pyplot(fig)

        # Classification report
        st.subheader("📋 Detailed Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        #st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']))
        
        #--------------
        try:
            # If jinja2 is installed, show styled table
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']))
        except AttributeError:
            # Otherwise, fallback to plain table
            st.dataframe(report_df)
            st.info("💡 Install `jinja2` for enhanced table styling: `pip install jinja2`")
        #-----------------

    except FileNotFoundError:
        st.error(f"❌ Model file '{model_file}' not found. Please run the training scripts first.")

else:
    st.info("👆 Upload a CSV file to evaluate the selected model.")
    st.markdown("""
    **Instructions:**
    1. First, run all `model/*_train.py` scripts to generate `.pkl` files.
    2. Upload `test_data.csv` (or any CSV with the same structure).
    3. The last column must be the target variable.
    """)