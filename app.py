import streamlit as st
import numpy as np    
import tensorflow as tf
import os
import urllib

# Importing necessary libraries for text processing
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model
model = tf.keras.models.load_model('D:/Masteriarv/s2/ml/Emotion-recognition-of-speech/Emotion-recognition-of-speech/mymodel.h5')

st.success("Text Emotion Recognition")

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Enter Text',)
    )
        
    if selected_box == 'Enter Text':        
        st.sidebar.success('Enter your text...')
        text_input()

def text_input():
    st.subheader("Enter Your Text:")
    comment_text = st.text_area("Input your comment here:")
    
    if st.button("Predict"):
        if comment_text:
            prediction = predict_emotion(comment_text)
            st.success(f"The text is predicted to be: {prediction}")
        else:
            st.warning("Please enter some text.")

def predict_emotion(text):
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
    )
    
    # Convert tokenized input to numpy arrays
    input_ids = np.array(inputs['input_ids'])
    attention_mask = np.array(inputs['attention_mask'])
    token_type_ids = np.array(inputs['token_type_ids'])
    
    # Make prediction
    prediction = model.predict([[input_ids], [attention_mask], [token_type_ids]])
    
    # Decode prediction
    emotion_labels = ['Neutral', 'Hate', 'Offensive']
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]
    
    return predicted_emotion

if __name__ == "__main__":
    main()
