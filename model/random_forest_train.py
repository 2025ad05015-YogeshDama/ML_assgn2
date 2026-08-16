# model/random_forest_train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_and_prepare_data
import os

os.makedirs('model', exist_ok=True)

X_train, X_test, y_train, y_test = load_and_prepare_data()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model/random_forest.pkl')
print("✅ Random Forest model saved as 'model/random_forest.pkl'")