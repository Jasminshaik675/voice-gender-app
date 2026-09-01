# Voice Gender Classification

## Project Description

Voice Gender Classification is a machine learning application that
predicts the gender of a speaker from an uploaded voice recording.

The project uses a Flutter frontend and a Flask backend.

The backend processes the audio, extracts MFCC features, and uses a
trained Random Forest machine learning model for classification.

## Technologies Used

- Python
- Flask
- Librosa
- NumPy
- Scikit-learn
- Joblib
- FFmpeg
- Flutter
- Dart

## How It Works

1. User opens the Flutter application.
2. User selects a voice recording.
3. Flutter sends the audio file to the Flask backend.
4. Flask receives the audio file.
5. Non-WAV files are converted to WAV using FFmpeg.
6. MFCC features are extracted from the audio.
7. The features are scaled.
8. The trained Random Forest model predicts the class.
9. The backend returns the prediction and confidence.
10. Flutter displays the result.

## Supported Audio Formats

- WAV
- MP3
- M4A
- AAC
- OGG

## Running the Backend

Open PowerShell and run:

    cd C:\Users\shaik\Downloads\flutter\flutter\voice_gender_app\backend

Then:

    python app.py

The backend runs on:

    http://127.0.0.1:5000

## Running the Flutter Application

Open another terminal:

    cd C:\Users\shaik\Downloads\flutter\flutter\voice_gender_app

Then run:

    flutter run -d windows

## Training the Model

To train the model:

    python train_model.py

The trained model files are:

- gender_model.pkl
- scaler.pkl

## API

### Endpoint

    POST /predict

### Upload Field

    audio

### Example Response

    {
      "filename": "sample.wav",
      "prediction": "Male",
      "confidence": 0.84
    }

## Prediction

The model predicts:

- Male
- Female

If the confidence is below the configured threshold, the application
displays:

    Unable to classify

## Project Structure

voice_gender_app/

    backend/
        app.py
        train_model.py
        gender_model.pkl
        scaler.pkl
        dataset/
            male/
            female/

    lib/
        main.dart

    pubspec.yaml
    README.md

## Testing

The application was tested using different voice recordings and
supported audio formats.

The complete pipeline is:

Flutter → Flask → Audio Processing → MFCC → Machine Learning Model
→ Prediction → Flutter UI