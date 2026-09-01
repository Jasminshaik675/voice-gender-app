from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import joblib

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gender_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Model loaded successfully.")
    print("Scaler loaded successfully.")

except Exception as e:
    print("ERROR loading model/scaler:")
    print(e)
    raise


# ============================================================
# EXTRACT AUDIO FEATURES
# ============================================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # --------------------------------------------------------
    # The trained model expects 40 features
    # --------------------------------------------------------

    features = mfcc_mean

    return features


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Voice Gender Classification API is running",
        "status": "success"
    })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether audio was uploaded
    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file received"
        }), 400

    audio_file = request.files["audio"]

    # Check filename
    if audio_file.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400

    file_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded audio
        # ----------------------------------------------------

        filename = os.path.basename(audio_file.filename)

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        audio_file.save(file_path)

        print("Received file:", filename)

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(file_path)

        print("Features extracted:", len(features))

        # ----------------------------------------------------
        # Convert to numpy array
        # ----------------------------------------------------

        features = np.array(features).reshape(1, -1)

        print("Feature shape:", features.shape)

        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        features_scaled = scaler.transform(features)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(features_scaled)[0]

        prediction = str(prediction)

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = 1.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features_scaled
            )[0]

            confidence = float(
                np.max(probabilities)
            )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        print(
            "Prediction:",
            prediction,
            "Confidence:",
            confidence
        )

        return jsonify({

            "prediction": prediction,

            "confidence": confidence

        }), 200

    except Exception as e:

        print("Prediction error:")
        print(e)

        return jsonify({

            "error": str(e)

        }), 500

    finally:

        # ----------------------------------------------------
        # Delete uploaded file after prediction
        # ----------------------------------------------------

        if file_path is not None:

            try:

                if os.path.exists(file_path):

                    os.remove(file_path)

            except Exception:

                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("==========================================")
    print("Voice Gender Classification API")
    print("==========================================")
    print("Starting Flask server...")
    print("Server URL: http://0.0.0.0:5000")
    print("==========================================")

if __name__ == "__main__":
    import os


    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )