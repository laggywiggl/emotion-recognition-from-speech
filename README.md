# 🎙️ Emotion Recognition from Speech

This project is a real-time application that detects human emotions (e.g., happy, sad, surprised) from voice recordings using deep learning and audio signal processing.

## 🚀 Features

- 🎧 Input: Speech audio files (WAV format)
- 🧠 Model: CNN + MFCC features trained on emotional datasets
- 📊 Technologies: Python, TensorFlow, Librosa, Streamlit
- 🗣️ Emotions Detected: Surprised, Disgusted, etc.

## 📁 Project Structure

📦 Emotion-recognition-of-speech/
├── app/ # Python app scripts
├── mymodel.h5 # Trained deep learning model
├── *.wav # Sample audio inputs
├── Streamlit_audio.ipynb # Jupyter notebook version
├── README.md

perl
Copier
Modifier

## 🛠 Installation

pip install -r requirements.txt
streamlit run app.py
🎤 Example
Try uploading a WAV file like Surprised.wav or record your own voive  and the app will return:

Detected Emotion: Surprised (example)

📚 Dependencies
Python 3.8+

TensorFlow

Librosa

Streamlit

NumPy, Scikit-learn
