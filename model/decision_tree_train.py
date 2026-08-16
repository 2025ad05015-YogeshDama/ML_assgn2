# model/decision_tree_train.py
import joblib
from sklearn.tree import DecisionTreeClassifier
from data_loader import load_and_prepare_data
import os

os.makedirs('model', exist_ok=True)

X_train, X_test, y_train, y_test = load_and_prepare_data()

# Decision Tree does not need scaling
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model/decision_tree.pkl')
print("✅ Decision Tree model saved as 'model/decision_tree.pkl'")