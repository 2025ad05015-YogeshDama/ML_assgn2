
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_and_prepare_data
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs('model', exist_ok=True)

# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler (only once – we do it here, but other scripts can skip)
joblib.dump(scaler, 'model/scaler.pkl')

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(model, 'model/logistic_regression.pkl')
print("✅ Logistic Regression model saved as 'model/logistic_regression.pkl'")