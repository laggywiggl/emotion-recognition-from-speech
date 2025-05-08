# 🎙️ Audio Emotion Recognition (SVM vs CNN)

This project compares two machine learning models — **Support Vector Machine (SVM)** and **Convolutional Neural Network (CNN)** — for recognizing emotions from speech audio files. It uses the **RAVDESS** dataset and includes a user-friendly interface built with **Streamlit**.

## 📌 Features

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
   git clone https://github.com/yourusername/audio-emotion-recognition.git
   cd audio-emotion-recognition
Install requirements:

bash
Copier
Modifier
pip install -r requirements.txt
Launch the app:

bash
Copier
Modifier
streamlit run app/interface.py
Upload a .wav file and get emotion predictions from both models!

📊 Results (Accuracy)
Model	Accuracy
SVM	97%
CNN	94%

👩‍💻 Authors
Zeghli Fatima Zahra

Ait Ouamer Ouafa

Bourass Wiame
