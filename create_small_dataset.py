
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
import unicodedata


def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

PATH = r'data_small/dev_wav'
onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

print(type(onlyfiles))
print(onlyfiles)

full_file_dev = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_development.csv', usecols=[0,1,2,3,4,5], encoding='utf-8-sig')
full_file_eval = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_evaluation.csv',  usecols =[0,1,2,3,4,5], encoding='utf-8-sig')
full_file = pd.concat([full_file_dev,full_file_eval])


for i in range(5):
    for j in range(full_file.shape[0]):
        full_file.iloc[j, i+1] = strip_accents_and_lowercase(full_file.iloc[j,i+1])
        full_file.iloc[j, i + 1] = re.sub(r'[.,"\'-?:!;]', '', full_file.iloc[j, i+1])
print(full_file.shape)

small_file = full_file

small_file = small_file[small_file['file_name'].isin(onlyfiles)]

print(small_file.shape)
#print(small_file)

#small_file.to_csv('small_dataset.csv', encoding='utf-8')

small_file_eval = small_file.sample(frac=0.2)
small_file_dev = small_file.drop(small_file_eval.index)


print(small_file_eval)
print('############')
print(small_file_dev)

#small_file_dev.to_csv('small_dataset_dev.csv', encoding='utf-8')
#small_file_eval.to_csv('small_dataset_eval.csv', encoding='utf-8')

print(onlyfiles)
print(full_file)

for i in range(5):
    for j in range(small_file_eval.shape[0]):
        small_file_eval.iloc[j, i+1] = small_file_dev.iloc[j, i+1]

small_file_eval.to_csv('small_dataset_eval22222.csv', encoding='utf-8')
small_file_dev.to_csv('small_dataset_dev2222222.csv', encoding = 'utf-8')

