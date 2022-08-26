from email.mime import audio
import os
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from CNN.SpeechEmotionRecognition import SpeechEmotionRecognition
st_audiorec = components.declare_component("st_audiorec", path=os.path.join(os.path.dirname(__file__), "st_audiorec/frontend/build"))

# Load Model

analyzer = SpeechEmotionRecognition(subdir_model = os.path.join(os.path.dirname(__file__), "./Model/CNN/MODEL_CNN_LSTM.hdf5"))

def predict_emotion_from_stream(audio_data, sample_rate):
    return analyzer.predict_emotion_from_stream(audio_data, sample_rate=sample_rate)

# UI

def visualizer(signal, sample_rate, predictions):
    fig, ax = plt.subplots()
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    plt.plot(time, signal)
    emotion_colors = {'Angry': 'red', 'Disgust': 'green', 'Fear': 'purple', 'Happy': 'yellow', 'Neutral' : 'grey', 'Sad': 'blue', 'Surprise': 'orange'}

    for i in range(len(predictions[1])):
        if i == 0:
            plt.axvspan(0, predictions[1][i], color = emotion_colors[predictions[0][i]], alpha=0.5, zorder=-100, label=predictions[0][i])
        else:
            plt.axvspan(predictions[1][i-1], predictions[1][i], color = emotion_colors[predictions[0][i]], alpha=0.5, zorder=-100, label=predictions[0][i])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    st.pyplot(fig)

def st_ui():

    st.write("# Audio Emotion Analysis 🎙️")
    st.markdown(
        """
        Extracting and analysing speech in an audio file has a wide variety of applications. 
        1) Automatic analysis and rating of call center calls. 
        2) Forensic analysis of audio and video files. 
        3) Emotion analysis for voice recognition systems. 
        """
    )

    audio_choice = st.radio(
     "Audio Source",
     ('Samples', 'Upload', 'Record'))

    if audio_choice == 'Samples':
        st.header("Try sample audio files!")
        sample_file = st.selectbox(
                    '',
                    ('Audio1', 'Happy', 'Angry', 'Sad', 'Fear'))
        file_path = os.path.join(os.path.dirname(__file__), "sample_audios/%s.wav" % (sample_file))
        st.audio(file_path)
        y, sr = analyzer.load_audio(file_path)
        btn = st.button("Generate")
        if btn:
            pred = analyzer.predict_emotion_from_stream(y, sample_rate=sr)
            visualizer(y, sr, pred)
            st.write(pred)
            
    elif audio_choice == 'Upload':
        uploaded_file = st.file_uploader("Choose a file", type=['wav', 'mp3'])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            y, sr = analyzer.load_audio(uploaded_file)

            btn = st.button("Generate")
            if btn:
                pred = analyzer.predict_emotion_from_stream(y, sample_rate=sr)
                visualizer(y, sr, pred)
                st.write(pred)
                
    else:
        val = st_audiorec()
        btn = st.button("Generate")
        if isinstance(val, dict) and btn:  # retrieve audio data
            with st.spinner('retrieving audio-recording...'):
                ind, val = zip(*val['arr'].items())
                ind = np.array(ind, dtype=int)  # convert to np array
                val = np.array(val)             # convert to np array
                sorted_ints = val[ind]
                stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                stream.seek(0)

                X, sample_rate = sf.read(stream)
                new_rate = 16000
                X = np.mean(X, axis=1)
                X = librosa.resample(X, orig_sr=sample_rate, target_sr=new_rate)
                pred = analyzer.predict_emotion_from_stream(X)
                visualizer(X, new_rate, pred)
                st.write(pred)

if __name__ == "__main__":
    st_ui()
    