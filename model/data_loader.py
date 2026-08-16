import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """Load UCI Heart Disease dataset, clean, split, and return train/test."""
    print("📥 Fetching Heart Disease dataset from UCI (ID=45)...")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Convert target to binary (0 = no disease, 1 = disease)
    y_binary = (y['num'] > 0).astype(int).rename('target')

    # Combine and drop missing values
    df = pd.concat([X, y_binary], axis=1)
    df = df.dropna()

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test data for the Streamlit app
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)
    print(f"✅ Test data saved as 'test_data.csv' ({len(test_df)} rows)")

    return X_train, X_test, y_train, y_test