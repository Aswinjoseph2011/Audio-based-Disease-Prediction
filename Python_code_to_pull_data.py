#https://www.kaggle.com/datasets/subhajournal/patient-health-detection-using-vocal-audio

'''install Libraries "librosa" --> pip install librosa  
'''

import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# filename = "patient-vocal-dataset-small/patient-vocal-dataset-small/Laryngozele/1205-a_h-egg.wav"
# plt.figure(figsize=(14,5))
# data,sample_rate=librosa.load(filename)
# librosa.display.waveshow(data,sr=sample_rate)
# ipd.Audio(filename)
# print(sample_rate)  ## by deafult librosa will give sample_rate of 22040

# from scipy.io import wavfile as wav
# wave_sample_rate,wave_audio = wav.read(filename)
# print(wave_sample_rate)

## Extracting MFCC's for every audio file
'''MFCC: 
      mel-frequency cepstral coefficients (MFCCs), are the final features used 
      in many machine learning models trained on audio data!'''

audio_dataset_path = "C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/merge_folder"
metadata = pd.read_csv('meta_data.csv')
print(metadata.head())

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path)+'\\',str(row["filename"]))
    final_class_labels=row["value"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])    

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
print(extracted_features_df.head())    

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

print(X.shape)
### Label Encoding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
print(y)

### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)

# Model Creation
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

### No of classes
num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.h5',
                               verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])
