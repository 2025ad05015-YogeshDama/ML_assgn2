import subprocess
import sys
import os

# List of training scripts (relative to project root)
scripts = [
    'model/logistic_regression_train.py',
    'model/decision_tree_train.py',
    'model/knn_train.py',
    'model/naive_bayes_train.py',
    'model/random_forest_train.py'
]

print("🚀 Starting training of all models...\n")

for script in scripts:
    print(f"▶️ Running: {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in {script}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print(f"✅ {script} completed successfully.\n")

print("🎉 All models trained successfully!")