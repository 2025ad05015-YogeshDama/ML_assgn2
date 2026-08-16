# model/knn_train.py
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from data_loader import load_and_prepare_data
import os

os.makedirs('model', exist_ok=True)

X_train, X_test, y_train, y_test = load_and_prepare_data()

# Scale features (KNN is distance-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# scaler already saved by LR script; we can skip saving again

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'model/knn.pkl')
print("✅ KNN model saved as 'model/knn.pkl'")