# Speech-Emotion-Recognition

Extracting and analysing speech in an audio file has a wide variety of applications. 

1) Automatic analysis and rating of call center calls. 
2) Forensic analysis of audio and video files. 
3) Emotion analysis for voice recognition systems. 

## Test API Call
```python
import os
import pydaisi as pyd
speech_emotion_recognition = pyd.Daisi("soul0101/Speech Emotion Recognition")

with open(os.path.join(os.path.dirname(__file__), "./test-rec.wav"), "rb") as f:
    audio_data, sample_rate = speech_emotion_recognition.get_audio_data(f).value

predictions = speech_emotion_recognition.predict_emotion_from_stream(audio_data, sample_rate).value
print(predictions)
```
