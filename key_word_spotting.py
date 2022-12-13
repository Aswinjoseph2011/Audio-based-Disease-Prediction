import librosa
import tensorflow as tf
import numpy as np
from voice_recorder import voice_record


SAVED_MODEL_PATH = "28_10_2022/saved_models/audio_classification.h5"
# filename="patient-vocal-dataset-small/patient-vocal-dataset-small/Vox senilis/816-a_h.wav"
filename="C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/recording.wav"
SAMPLES_TO_CONSIDER = 22050

voice  = voice_record()
file = voice.record()
print("Audio Data is succes")

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """
    model = None
    _mapping = [
        "Laryngozele",
        "Normal",
        "Vox_senilis"
    ]
    _instance = None


    def predict(self, filename):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """
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

a = _Keyword_Spotting_Service()
#a.Table_Adverse_event()
keyword = a.predict(filename)
print(f"file: {filename}, \n prediction: {keyword}")