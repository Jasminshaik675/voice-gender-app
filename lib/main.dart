import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const VoiceGenderApp());
}

class VoiceGenderApp extends StatelessWidget {
  const VoiceGenderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Voice Gender Classification',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? selectedFileName;
  String? prediction;
  double? confidence;
  String? errorMessage;

  bool isLoading = false;

  Future<void> uploadAudio() async {
    try {
      // Select audio file
      final PlatformFile? file = await FilePicker.pickFile(
        type: FileType.custom,
        allowedExtensions: [
          'wav',
          'mp3',
          'm4a',
          'aac',
          'ogg',
        ],
      );

      // User cancelled
      if (file == null) {
        return;
      }

      // Read audio bytes
      final bytes = await file.readAsBytes();

      if (bytes.isEmpty) {
        throw Exception(
          'Unable to read the selected audio file.',
        );
      }

      setState(() {
        selectedFileName = file.name;
        prediction = null;
        confidence = null;
        errorMessage = null;
        isLoading = true;
      });

      // Android emulator:
      // 10.0.2.2 points to your computer localhost.
      final request = http.MultipartRequest(
        'POST',
       Uri.parse(
  'https://voice-gender-app-1.onrender.com/predict',
),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'audio',
          bytes,
          filename: file.name,
        ),
      );

      // Send audio to Flask
      final response = await request.send();

      final responseBody =
          await response.stream.bytesToString();

      final dynamic decodedData =
          jsonDecode(responseBody);

      if (!mounted) {
        return;
      }

      if (response.statusCode == 200) {
        final Map<String, dynamic> data =
            Map<String, dynamic>.from(decodedData);

        // Get prediction from backend
        final String modelPrediction =
            data['prediction'].toString();

        // IMPORTANT:
        // Your model returns:
        // 0 = Male
        // 1 = Female
        String gender;

        if (modelPrediction == '0') {
          gender = 'Male';
        } else if (modelPrediction == '1') {
          gender = 'Female';
        } else {
          gender = modelPrediction;
        }

        setState(() {
          prediction = gender;

          confidence =
              (data['confidence'] as num).toDouble();

          isLoading = false;
          errorMessage = null;
        });
      } else {
        final Map<String, dynamic> data =
            Map<String, dynamic>.from(decodedData);

        throw Exception(
          data['error']?.toString() ??
              'Prediction failed.',
        );
      }
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        isLoading = false;
        errorMessage = e.toString();
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Prediction failed: $e',
          ),
        ),
      );
    }
  }

  Color getPredictionColor() {
    if (prediction == 'Male') {
      return Colors.blue;
    }

    if (prediction == 'Female') {
      return Colors.pink;
    }

    return Colors.orange;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F8FF),

      appBar: AppBar(
        title: const Text(
          'Voice Gender Classification',
        ),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),

      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),

          child: Column(
            mainAxisAlignment:
                MainAxisAlignment.center,

            children: [
              // =========================
              // MICROPHONE ICON
              // =========================

              const Icon(
                Icons.mic,
                size: 100,
                color: Colors.blue,
              ),

              const SizedBox(height: 20),

              // =========================
              // TITLE
              // =========================

              const Text(
                'Voice Gender Classification',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 15),

              // =========================
              // DESCRIPTION
              // =========================

              const Text(
                'Upload an audio file to predict '
                'the speaker gender.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 16,
                ),
              ),

              const SizedBox(height: 40),

              // =========================
              // UPLOAD BUTTON
              // =========================

              SizedBox(
                width: 220,
                height: 50,

                child: ElevatedButton.icon(
                  onPressed:
                      isLoading ? null : uploadAudio,

                  icon: const Icon(
                    Icons.upload_file,
                  ),

                  label: Text(
                    isLoading
                        ? 'Processing...'
                        : 'Upload Audio',

                    style: const TextStyle(
                      fontSize: 16,
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 30),

              // =========================
              // LOADING
              // =========================

              if (isLoading)
                const Column(
                  children: [
                    CircularProgressIndicator(),

                    SizedBox(height: 15),

                    Text(
                      'Analyzing voice...',
                      style: TextStyle(
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),

              // =========================
              // SELECTED FILE
              // =========================

              if (selectedFileName != null &&
                  !isLoading) ...[
                const Icon(
                  Icons.audio_file,
                  size: 50,
                  color: Colors.green,
                ),

                const SizedBox(height: 10),

                const Text(
                  'Selected file:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 5),

                Text(
                  selectedFileName!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 16,
                    color: Colors.blue,
                  ),
                ),

                const SizedBox(height: 25),
              ],

              // =========================
              // PREDICTION
              // =========================

              if (prediction != null &&
                  confidence != null &&
                  !isLoading)

                Container(
                  width: double.infinity,

                  padding: const EdgeInsets.all(20),

                  decoration: BoxDecoration(
                    borderRadius:
                        BorderRadius.circular(15),

                    border: Border.all(
                      color: getPredictionColor(),
                      width: 2,
                    ),
                  ),

                  child: Column(
                    children: [
                      const Text(
                        'Prediction',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),

                      const SizedBox(height: 15),

                      // =========================
                      // MALE / FEMALE
                      // =========================

                      Text(
                        prediction!,
                        textAlign: TextAlign.center,

                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                          color:
                              getPredictionColor(),
                        ),
                      ),

                      const SizedBox(height: 15),

                      // =========================
                      // CONFIDENCE
                      // =========================

                      Text(
                        'Confidence: '
                        '${(confidence! * 100).toStringAsFixed(1)}%',

                        style: const TextStyle(
                          fontSize: 18,
                        ),
                      ),
                    ],
                  ),
                ),

              // =========================
              // ERROR
              // =========================

              if (errorMessage != null)
                Container(
                  margin:
                      const EdgeInsets.only(top: 20),

                  padding:
                      const EdgeInsets.all(15),

                  decoration: BoxDecoration(
                    borderRadius:
                        BorderRadius.circular(10),

                    border: Border.all(
                      color: Colors.red,
                    ),
                  ),

                  child: Text(
                    errorMessage!,
                    textAlign: TextAlign.center,

                    style: const TextStyle(
                      color: Colors.red,
                      fontSize: 15,
                    ),
                  ),
                ),

              // =========================
              // INITIAL MESSAGE
              // =========================

              if (selectedFileName == null &&
                  prediction == null &&
                  !isLoading)

                const Padding(
                  padding:
                      EdgeInsets.only(top: 20),

                  child: Text(
                    'Prediction will appear here',

                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.grey,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}