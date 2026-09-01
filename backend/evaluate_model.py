import os
import numpy as np
import librosa
import joblib

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
# Load trained model and scaler
# --------------------------------------------------

print("Loading model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully.")
print("Scaler loaded successfully.")

# --------------------------------------------------
# Extract MFCC features
# --------------------------------------------------

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=None
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        features = np.mean(
            mfcc,
            axis=1
        )

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

X = []
y = []

print("\nLoading dataset...")

class_names = {
    "male": 0,
    "female": 1
}

for class_name, label in class_names.items():

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(f"\nProcessing {class_name} files...")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(
            (".wav", ".mp3", ".m4a", ".aac", ".ogg")
        ):
            continue

        file_path = os.path.join(
            folder_path,
            filename
        )

        features = extract_features(
            file_path
        )

        if features is not None:
            X.append(features)
            y.append(label)

print("\nDataset loading completed.")

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# --------------------------------------------------
# Scale features
# --------------------------------------------------

print("\nScaling features...")

X_scaled = scaler.transform(X)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

print("Making predictions...")

y_pred = model.predict(X_scaled)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(y, y_pred)
)

print("\nClassification Report:")
print(
    classification_report(
        y,
        y_pred,
        target_names=["Male", "Female"],
        zero_division=0
    )
)

print("=" * 50)
print("Evaluation completed successfully.")