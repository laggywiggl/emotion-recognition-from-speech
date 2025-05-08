import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import streamlit as st
import numpy as np    
import tensorflow as tf
import os,urllib
from audio_recorder_streamlit import audio_recorder

import librosa # to extract speech features


__login__obj = __login__(auth_token = "courier_auth_token", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:

    st.success("Speech emotion recognition ")
        
    def main():
        #print(cv2.__version__)
        selected_box = st.sidebar.selectbox(
            'Choose an option..',
            ('Upload audio','Record audio')
            )
                
        if selected_box == 'Upload audio':        
            st.sidebar.success('Upload audio file...')
            application()
        if selected_box == 'Record audio':        
            st.sidebar.success('Record audio please...')
            audio()
       

    
  
    
    def load_model():
        model=tf.keras.models.load_model('D:/Masteriarv/s2/ml/Emotion-recognition-of-speech/Emotion-recognition-of-speech/mymodel.h5')
        
        return model
    def application():
        models_load_state=st.text('\n Loading models..')
        model=load_model()
        models_load_state.text('\n Models Loading..complete')
        
        
        file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
        
        if file_to_be_uploaded:
            st.audio(file_to_be_uploaded, format='audio/wav')
            st.success('Emotion of the audio is  '+predict(model,file_to_be_uploaded))

    def audio():
        #models_load_state=st.text('\n Loading models..')
        model=load_model()
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            #print(predict(model,audio_bytes))
            #st.success('Emotion of the audio is  '+predict(model,audio_bytes))


    def extract_mfcc(wav_file_name):
        #This function extracts mfcc features and obtain the mean of each dimension
        #Input : path_to_wav_file
        #Output: mfcc_features'''
        y, sr = librosa.load(wav_file_name)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
        
        return mfccs
        
        
    def predict(model,wav_filepath):
        emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
        test_point=extract_mfcc(wav_filepath)
        test_point=np.reshape(test_point,newshape=(1,40,1))
        #predict and print result
        predictions=model.predict(test_point)
        print(emotions[np.argmax(predictions[0])+1])
        
        return emotions[np.argmax(predictions[0])+1]
    if __name__ == "__main__":
        main()
