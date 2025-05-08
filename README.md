# 🎙️ Audio Emotion Recognition (SVM vs CNN)

This project compares two machine learning models — **Support Vector Machine (SVM)** and **Convolutional Neural Network (CNN)** — for recognizing emotions from speech audio files. It uses the **RAVDESS** dataset and includes a user-friendly interface built with **Streamlit**.

## 📌 Features
![Streaming App UI](prediction.jpg)
- Emotion detection from speech (e.g., happy, sad, angry, etc.)
- Audio preprocessing and feature extraction (MFCC, Chroma, etc.)
- SVM model using handcrafted features
- CNN model trained on MFCC spectrograms
- Streamlit app for real-time testing

## 🗂️ Project Structure

├── app/ # Streamlit interface

├── data/ # Audio files (RAVDESS)

├── models/ # Saved models (SVM .pkl & CNN .h5)

├── notebooks/ # Jupyter notebooks for training

├── requirements.txt # Python dependencies

└── README.md



## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/laggywiggl/audio-emotion-recognition.git
   cd audio-emotion-recognition
2. Install requirements:
   ```bash
   pip install -r requirements.txt
4. Launch the app:
   ```bash
   streamlit run app.py
   
Upload a .wav file and get emotion predictions from both models!

## 📊 Results (Accuracy)
Model	Accuracy
SVM	97%
CNN	94%
