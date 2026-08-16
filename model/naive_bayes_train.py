# model/naive_bayes_train.py
import joblib
from sklearn.naive_bayes import GaussianNB
from data_loader import load_and_prepare_data
import os

os.makedirs('model', exist_ok=True)

X_train, X_test, y_train, y_test = load_and_prepare_data()

model = GaussianNB()
model.fit(X_train, y_train)   # Naive Bayes works with raw features

joblib.dump(model, 'model/naive_bayes.pkl')
print("✅ Naive Bayes model saved as 'model/naive_bayes.pkl'")