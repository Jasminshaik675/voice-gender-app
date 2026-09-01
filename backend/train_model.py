import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_PATH = "dataset"
MODEL_PATH = "gender_model.pkl"
SCALER_PATH = "scaler.pkl"

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        features = np.mean(mfcc, axis=1)

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

X = []
y = []

class_names = {
    "male": 0,
    "female": 1
}

print("Loading dataset...")

for class_name, label in class_names.items():

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(f"\nProcessing {class_name} files...")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".wav"):
            continue

        file_path = os.path.join(
            folder_path,
            filename
        )

        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(label)

print("\nDataset loading completed.")

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", np.bincount(y))

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train
)

# --------------------------------------------------
# Test prediction
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(
    X_test_scaled
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Male", "Female"],
        zero_division=0
    )
)

print("=" * 50)

# --------------------------------------------------
# Save model and scaler
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\nTraining completed!")
print("Model saved as gender_model.pkl")
print("Scaler saved as scaler.pkl")