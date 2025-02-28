import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load labeled data from separate CSVs
file_paths = ["C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\BDI_P_oscillation.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\BDI_P_overlimit.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\BDI_Q_oscillation.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\BDI_Q_overlimit.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\FDI_P.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\FDI_SOC.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\FIRMWARE_THD_modification.csv",
    "C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\Dataset\\FIRMWARE_Voltage_modification.csv"]

labeled_data = []
for file in file_paths:
    data = pd.read_csv(file)
    labeled_data.append(data)

labeled_df = pd.concat(labeled_data, axis=0)

# Separate features and labels
X = labeled_df.drop(columns=['label'])  # Assuming 'label' column contains 0 or 1
y = labeled_df['label']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split labeled data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate on validation data
val_predictions = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
conf_matrix = confusion_matrix(y_val, val_predictions)

print("Validation Accuracy:", accuracy)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_val, val_predictions))

# Load and predict on unlabeled normal data
unlabeled_df = pd.read_csv(r"C:\\Users\\prajw\\Desktop\\EOC-BESS\\BESS-Set\\Bess_Set\\training.csv")  # Adjust path
X_unlabeled = scaler.transform(unlabeled_df)

predictions = xgb_model.predict(X_unlabeled)
unlabeled_df['predictions'] = predictions

# Save predictions
unlabeled_df.to_csv('predictions_output.csv', index=False)
print("Predictions saved to 'predictions_output.csv'")

