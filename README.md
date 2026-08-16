# Last Update Date : 16/8/2026


🚀 Getting Started

1. Prerequisites

Python 3.8 or higher
pip package manager
Git (to clone the repository)

2. Clone the Repository

$ git clone https://github.com/your-username/heart-disease-assignment.git
$ cd heart-disease-assignment

3. Create a Virtual Environment (Recommended)

$ python -m venv venv
$ source venv/bin/activate      # Linux/Mac
# 			or
$ venv\Scripts\activate         # Windows

4. Install Dependencies

$ pip install -r requirements.txt



🖥️ Running the Streamlit App

$ streamlit run app.py

🧪 Using the App
1. Upload Test Data
Click "Choose a CSV file" and upload any CSV that has:

All 13 feature columns (in any order)

A target column as the last column (0 = no disease, 1 = disease)

If you haven't trained the models yet, the app will generate test_data.csv automatically during the first training.

2. Select a Model
In the sidebar, choose from:
	Logistic Regression
	Decision Tree
	KNN
	Naive Bayes
	Random Forest

3. Train & Evaluate the Selected Model
Click the "⚙️ Train & Evaluate Selected" button in the sidebar.

If the model is not yet trained, the app will run the corresponding training script (this may take a few seconds).

Once trained (or if already present), the model is loaded and evaluated on your uploaded data.

4. View Results
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


📎 Live Deployment
The app is deployed on Streamlit Community Cloud and accessible at:

🔗 Streamlit App Link here : 

🤝 Contributing
This project was developed as part of the M.Tech (AIML) curriculum at BITS Pilani.
For any issues or suggestions, please open an issue on GitHub.

📜 License
This project is for educational purposes only. The dataset is from the UCI Machine Learning Repository and is used under its license terms.

