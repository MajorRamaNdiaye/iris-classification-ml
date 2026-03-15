# Iris Flower Classification using Decision Tree
# Author: Major Rama Ndiaye
# Dataset: Iris (sklearn built-in)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0=setosa, 1=versicolor, 2=virginica

print(f"Dataset: Iris")
print(f"Total samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {iris.target_names.tolist()}")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Train Decision Tree model
# max_depth=3 chosen over default (overfitting) and depth=1 (underfitting)
model = DecisionTreeClassifier(
    max_depth=3,
    criterion='gini',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Results ---")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print(f"\nFeature Importance:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.4f}")
