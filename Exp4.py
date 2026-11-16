import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Datasets/housing.csv")
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_no_scale = LogisticRegression(max_iter=5000, solver='lbfgs')
model_no_scale.fit(X_train, y_train)
y_pred_no_scale = model_no_scale.predict(X_test)

print("Without Scaling:")
print("Accuracy:", accuracy_score(y_test, y_pred_no_scale))
print("Precision:", precision_score(y_test, y_pred_no_scale, average='macro'))
print("Recall:", recall_score(y_test, y_pred_no_scale, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_no_scale, average='macro'))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_no_scale)
plt.title("Confusion Matrix (Without Scaling)")
plt.show()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=5000, solver='lbfgs')
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

print("\nWith Scaling:")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
print("Precision:", precision_score(y_test, y_pred_scaled, average='macro'))
print("Recall:", recall_score(y_test, y_pred_scaled, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_scaled, average='macro'))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_scaled)
plt.title("Confusion Matrix (With Scaling)")
plt.show()
