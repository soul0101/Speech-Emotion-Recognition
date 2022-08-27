import os
import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from CNN.SpeechEmotionRecognition import SpeechEmotionRecognition
# Load Model
analyzer = SpeechEmotionRecognition(subdir_model = os.path.join(os.path.dirname(__file__), "./Model/CNN/MODEL_CNN_LSTM.hdf5"))

def predict_emotion_from_stream(audio_data : np.ndarray, sample_rate : int = 16000 ) -> list:
    """
    Gets the emotion predictions for the audio track

    Parameters
    ----------
    audio_data: np.ndarray [shape=(n,) or (…, n)]
        audio time series. Multi-channel is supported
    sample_rate: int
        sampling rate of the audio_data

    Returns
    -------
    predictions: list [emotions, timestamp]
        List of predicted emotions and timestamps
    """

    predictions = analyzer.predict_emotion_from_stream(audio_data, sample_rate=sample_rate)
    return predictions

def get_audio_data(file_object):
    """
    Gets audio time series from python file object

    Parameters
    ----------
    file_object: python file object opened in 'rb' mode

    Returns
    -------
    audio_data: np.ndarray [shape=(n,) or (…, n)]
        audio time series. Multi-channel is supported
    sample_rate: int
        sampling rate of the audio_data
    """
    return file_object
    extension = os.path.splitext(file_object.name)[1]
    if extension not in [".wav"]:
        raise TypeError("This file type is not supported, please only use 'wav' files")

    audio_data, sample_rate = librosa.load(file_object, sr=16000)
    return audio_data, sample_rate
    
def visualizer(audio_data : np.ndarray, sample_rate : int, predictions : list) -> matplotlib.figure.Figure: 
    """
    Plots audio spectrogram with predictions

    Parameters
    ----------
    audio_data: np.ndarray [shape=(n,) or (…, n)]
        audio time series. Multi-channel is supported.
    sample_rate: int
        sampling rate of the audio_data
    predictions: list [emotions, timestamp]
        List of predicted emotions and timestamps

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots()
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.plot(time, audio_data)
    emotion_colors = {'Angry': 'red', 'Disgust': 'green', 'Fear': 'purple', 'Happy': 'yellow', 'Neutral' : 'grey', 'Sad': 'blue', 'Surprise': 'orange'}

    for i in range(len(predictions[1])):
        if i == 0:
            plt.axvspan(0, predictions[1][i], color = emotion_colors[predictions[0][i]], alpha=0.5, zorder=-100, label=predictions[0][i])
        else:
            plt.axvspan(predictions[1][i-1], predictions[1][i], color = emotion_colors[predictions[0][i]], alpha=0.5, zorder=-100, label=predictions[0][i])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    return fig

################################## UI ##############################################

def predictions_table(predictions):
    data = {
        'Time Stamp (s)': predictions[1],
        'Emotion': predictions[0]
    }
    return pd.DataFrame(data)

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
            pred = predict_emotion_from_stream(y, sample_rate=sr)
            fig = visualizer(y, sr, pred)
            st.pyplot(fig)
            st.dataframe(predictions_table(pred))
            
    elif audio_choice == 'Upload':
        uploaded_file = st.file_uploader("Choose a file", type=['wav'])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            y, sr = analyzer.load_audio(uploaded_file)

            btn = st.button("Generate")
            if btn:
                pred = predict_emotion_from_stream(y, sample_rate=sr)
                fig = visualizer(y, sr, pred)
                st.pyplot(fig)
                st.dataframe(predictions_table(pred))
                
    else:
        st_audiorec = components.declare_component("st_audiorec", path=os.path.join(os.path.dirname(__file__), "st_audiorec/frontend/build"))
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
                pred = predict_emotion_from_stream(X, sample_rate = sample_rate)
                fig = visualizer(X, sample_rate, pred)
                st.pyplot(fig)
                st.dataframe(predictions_table(pred))

if __name__ == "__main__":
    st_ui()
    