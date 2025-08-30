import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load Iris dataset
iris = load_iris()

# 2. Create DataFrame with proper column names
df = pd.DataFrame(
    data=iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)
df["target"] = iris.target

# 3. Split dataset
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Save trained pipeline at your location
save_path = r"E:\Udemy\AzureDeployment\Model"
os.makedirs(save_path, exist_ok=True)  # create folder if it doesn't exist
model_file = os.path.join(save_path, "model.pkl")
joblib.dump(clf, model_file)

print(f"✅ Model saved to {model_file}")
