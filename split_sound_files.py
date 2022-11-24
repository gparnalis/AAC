import shutil
import os
import glob
import csv
import pandas as pd

new_dev = pd.read_csv(r"..\removed_extra_words_cut_DEV.csv", encoding= 'utf-8-sig')
new_eval= pd.read_csv(r"..\removed_extra_words_cut_EVAL.csv", encoding='utf-8-sig')
print(new_eval)
print(new_eval.shape[0])
list_of_names = []
for indx in range(new_eval.shape[0]):
    list_of_names.append(new_eval.iloc[indx,1])
print(list_of_names)

print(list_of_names[4])
print(len(list_of_names))
dest_path = r'../dataset/new_data_split/clotho_audio_files/evaluation'
source_dir = r'../dataset/new_data_split/clotho_audio_files'
target_dir = r'../dataset/new_data_split/clotho_audio_files/evaluation'
file_names = os.listdir(source_dir)
print(type(file_names))
print(len(file_names))
for file in file_names:
    if file in list_of_names:
        shutil.move(os.path.join(source_dir, file), target_dir)
