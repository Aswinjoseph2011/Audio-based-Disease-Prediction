from distutils.errors import DistutilsTemplateError
from typing import Optional
from fastapi import FastAPI
from enum import Enum
from starlette.status import HTTP_200_OK, HTTP_403_FORBIDDEN, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from fastapi import APIRouter, Query, HTTPException
import librosa
import tensorflow as tf
import numpy as np
from voice_recorder import voice_record





app = FastAPI()

@app.get("/Vocal_Data_Classifier/",
         status_code= HTTP_200_OK,
         name="GET:Data"
         )

# 
# # filename="patient-vocal-dataset-small/patient-vocal-dataset-small/Vox senilis/816-a_h.wav"
# 
# SAMPLES_TO_CONSIDER = 22050

# voice  = voice_record()
# file = voice.record()
# model = None
# _mapping = [
#     "Laryngozele",
#     "Normal",
#     "Vox_senilis"
#     ]
#  _instance = None


def predict(self):
    """
    :param file_path (str): Path to audio file to predict
    :return predicted_keyword (str): Keyword predicted by the model
    """
    SAVED_MODEL_PATH = "saved_models/audio_classification.h5"
    filename="C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/recording.wav"
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    predicted_label=model.predict(mfccs_scaled_features)
    classes_x=np.argmax(predicted_label,axis=1)
    print(classes_x)
    for i in classes_x:
        classes_x = i
    # get the predicted label
    predicted_keyword = self._mapping[classes_x]
    return predicted_keyword
    connection.close()
    engine.dispose()
    return "success"          