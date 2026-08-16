# ❤️ Heart Disease Classification – Model Comparison Dashboard

## a. Problem Statement
Heart disease is one of the leading causes of death worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. The goal of this project is to build a **machine learning classification system** that predicts whether a patient has heart disease based on **13 clinical features**.


## b. Dataset Description

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository – Heart Disease Dataset (ID: 45) |
| **URL** | https://archive.ics.uci.edu/dataset/45/heart+disease |
| **Instances** | 303 (after removing missing values) |
| **Features** | 13 |
| **Target** | Binary (0 = No Disease, 1 = Disease) |


## b. Dataset Description

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository – Heart Disease Dataset (ID: 45) |
| **URL** | https://archive.ics.uci.edu/dataset/45/heart+disease |
| **Instances** | 303 (after removing missing values) |
| **Features** | 13 |
| **Target** | Binary (0 = No Disease, 1 = Disease) |

### Feature List

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `age` | Age in years | Numeric |
| 2 | `sex` | 1 = male, 0 = female | Binary |
| 3 | `cp` | Chest pain type (0–3) | Ordinal |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| 5 | `chol` | Serum cholesterol (mg/dl) | Numeric |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| 7 | `restecg` | Resting ECG results (0–2) | Ordinal |
| 8 | `thalach` | Maximum heart rate achieved | Numeric |
| 9 | `exang` | Exercise induced angina (1 = yes) | Binary |
| 10 | `oldpeak` | ST depression induced by exercise | Numeric |
| 11 | `slope` | Slope of peak exercise ST segment | Ordinal |
| 12 | `ca` | Number of major vessels (0–3) | Ordinal |
| 13 | `thal` | Thalassemia (1–3) | Ordinal |
| **14** | **`target`** | **0 = No Disease, 1 = Disease** | **Target** |

### Dataset Statistics
- **Total Instances:** 303
- **No Disease (0):** 164 (54.1%)
- **Disease (1):** 139 (45.9%)
- **Train/Test Split:** 80% / 20% (stratified)
- **Test Set Size:** 60 instances (saved as `test_data.csv`)

---

## c. GitHub Repository Link

🔗 **Repository:** [https://github.com/2025ad05015-YogeshDama/ML_assgn2](https://github.com/2025ad05015-YogeshDama/ML_assgn2)

> *Replace with your actual GitHub repository URL after creating it.*

The repository contains:
- Complete source code (`app.py`, `train_all.py`, `data_loader.py`)
- Individual model training scripts (in `model/` folder)
- Saved model files (`.pkl` files)
- `requirements.txt`
- `README.md`
- `test_data.csv` (auto‑generated)


## d. Models Used – Performance Comparison

The following **5 classification models** were implemented and evaluated on the same test set. All metrics were computed on the **20% hold‑out test data**.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | **0.8525** | **0.9187** | **0.8400** | **0.8235** | **0.8317** | **0.7043** |
| Decision Tree | 0.7869 | 0.7812 | 0.7714 | 0.7826 | 0.7770 | 0.5734 |
| KNN | 0.8197 | 0.8725 | 0.8000 | 0.8125 | 0.8062 | 0.6405 |
| Naive Bayes | 0.8361 | 0.8903 | 0.8276 | 0.8276 | 0.8276 | 0.6722 |
| Random Forest | **0.8689** | **0.9321** | **0.8571** | **0.8571** | **0.8571** | **0.7379** |



## e. Observations on Model Performance

| ML Model | Observation |
|----------|-------------|
| **Logistic Regression** | Strong linear baseline. Performs well with scaled features. Achieves good AUC (0.92), indicating excellent class separation. Fastest to train. Interpretable coefficients. |
| **Decision Tree** | Highly interpretable and captures non‑linear patterns. Prone to overfitting (gap between train/test performance). Handles mixed data types well without scaling. Lower MCC suggests weaker separation. |
| **KNN** | Instance‑based learning. Sensitive to feature scaling and choice of `k`. Works well with local patterns but slower at prediction time. Balanced precision/recall. |
| **Naive Bayes** | Probabilistic classifier. Assumes feature independence – works surprisingly well on this dataset. Very fast training and inference. Handles high‑dimensional data well. |
| **Random Forest** | **Overall winner.** Ensemble of decision trees. Reduces overfitting through bagging and feature randomness. Consistently high across all metrics. Best AUC (0.93) and MCC (0.74). Most robust model. |

### Overall Winner
**🏆 Random Forest** performed best on this dataset with:
- Accuracy: **86.89%**
- F1 Score: **0.8571**
- AUC: **0.9321**
- MCC: **0.7379**

---

## f. How to Reproduce These Results

### 🚀 Getting Started*
**Last Update Date : 16/8/2026**
### 1. Prerequisites

	Python 3.8 or higher
	
	pip package manager
	
	Git (to clone the repository)

## 2. Clone the Repository

	git clone https://github.com/2025ad05015-YogeshDama/ML_assgn2.git
	cd ML_assgn2

## 3. Create a Virtual Environment (Recommended)

	python -m venv venv
	source venv/bin/activate      # Linux/Mac
	# 			or
	venv\Scripts\activate         # Windows

## 4. Install Dependencies

	pip install -r requirements.txt


### 🖥️ Running the Streamlit App

	streamlit run app.py

#### 🧪 Using the App
## 1. Upload Test Data
	Click "Choose a CSV file" and upload any CSV that has:
	All 13 feature columns (in any order)
	A target column as the last column (0 = no disease, 1 = disease)

	If you haven't trained the models yet, the app will generate test_data.csv automatically during the first training.

## 2. Select a Model

	In the sidebar, choose from:
		Logistic Regression
		Decision Tree
		KNN
		Naive Bayes
		Random Forest

## 3. Train & Evaluate the Selected Model

	Click the "⚙️ Train & Evaluate Selected" button in the sidebar.
	
	If the model is not yet trained, the app will run the corresponding training script (this may take a few seconds).
	
	Once trained (or if already present), the model is loaded and evaluated on your uploaded data.

## 4. View Results
	The app displays:
		Performance Metrics (6 metrics):
		Accuracy
		AUC (Area Under the ROC Curve)
		Precision
		Recall
		F1 Score
		Matthews Correlation Coefficient (MCC)
		Confusion Matrix – a heatmap showing correct/incorrect predictions.
		Detailed Classification Report – precision, recall, f1‑score per class.

## 📎 Live Deployment
	The app is deployed on Streamlit Community Cloud and accessible at:

	🔗 Streamlit App Link here : https://mlassgn2-yqpngcj34v8dx4qyckbmnq.streamlit.app/

## 🤝 Contributing

	This project was developed as part of the M.Tech (AIML) curriculum at BITS Pilani.
For any issues or suggestions, please open an issue on GitHub.

## 📜 License

	This project is for educational purposes only. The dataset is from the UCI Machine Learning Repository and is used under its license terms.

