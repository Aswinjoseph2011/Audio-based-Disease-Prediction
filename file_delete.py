import os,glob
folder = r'C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/patient-vocal-dataset/patient-vocal-dataset/Vox senilis' # change the folder name
for file_name in glob.os.listdir(folder):
    if "-egg" in file_name:
        os.remove(f"C:/Users/Aswinjoseph/Documents/AI_FOR_SOUND/28_10_2022/patient-vocal-dataset/patient-vocal-dataset/Vox senilis/{file_name}") # change the folder name 
        print("success")
