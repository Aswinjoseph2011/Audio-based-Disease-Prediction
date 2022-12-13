import os,glob
import pandas as pd


folder = r'C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/patient-vocal-dataset/patient-vocal-dataset/Laryngozele' # change the folder name
list = []
for file_name in glob.os.listdir(folder):
    list.append(file_name)
Laryngozele = pd.DataFrame(list,columns=["filename"])  
Laryngozele["value"]  = 'Laryngozele'
print(Laryngozele.head())

folder = r'C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/patient-vocal-dataset/patient-vocal-dataset/Normal' # change the folder name
list = []
for file_name in glob.os.listdir(folder):
    list.append(file_name)
Normal = pd.DataFrame(list,columns=["filename"])  
Normal["value"]  = 'Normal'
print(Normal.head())

folder = r'C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/patient-vocal-dataset/patient-vocal-dataset/Vox senilis' # change the folder name
list = []
for file_name in glob.os.listdir(folder):
    list.append(file_name)
Vox_senilis = pd.DataFrame(list,columns=["filename"])  
Vox_senilis["value"]  = 'Vox_senilis'
print(Vox_senilis.head())

# concatenating all dataframe
concat = pd.concat([Laryngozele, Normal,Vox_senilis], axis=0)
# saving the dataframe as csv
concat.to_csv('C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/meta_data.csv',index=False)