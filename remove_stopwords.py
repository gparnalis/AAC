import pandas as pd
import numpy as np

development_split = pd.read_csv(r'/Users/george/Documents/python/greek/after/dev_final.csv', encoding='utf-8-sig')
evaluation_split = pd.read_csv(r'/Users/george/Documents/python/greek/after/eval_final.csv', encoding='utf-8-sig')

development_split2 = pd.read_csv(r'/Users/george/Documents/python/greek/after/dev_final.csv', encoding='utf-8-sig')
evaluation_split2 = pd.read_csv(r'/Users/george/Documents/python/greek/after/eval_final.csv', encoding='utf-8-sig')


print(development_split.shape)
print(evaluation_split.shape)

a = development_split.caption_1.str.split(expand=True).stack().value_counts()
print(a)
b2 = development_split.caption_2.str.split(expand=True).stack().value_counts()
print(b2)
b3 = development_split.caption_3.str.split(expand=True).stack().value_counts()
print(b3)
b4 = development_split.caption_4.str.split(expand=True).stack().value_counts()
print(b4)
b5 = development_split.caption_5.str.split(expand=True).stack().value_counts()
print(b5)
print(type(a))
print("#########################################")
print(a[0:10])
print(b2[0:10])
print(b3[0:10])
print(b4[0:10])
print(b5[0:10])
print("#########################################")


stop_words = ["και", "σε", "ενα", "ενας", "μια", "αν", "σε", "καθως", "απο", "να", "στο", "του", "με", "που", 'αλλα',
              "αντι", "αυτα", "αυτες", "αυτοι", "ειναι", "ενω", "μα", "ο", "η", "το", "τα", "την", "οι", "στην", "στη",
              "στον", "εναν", "τον"]
for i in range(6):
    for j in range(development_split.shape[0]):  # shape: (n, 6)
        string_list = development_split.iloc[j, i+1].split()
        final_list = [word for word in string_list if word not in stop_words]
        development_split.iloc[j, i+1] = ' '.join(final_list)

for i in range(6):
    for j in range(evaluation_split.shape[0]):  # shape: (n, 6)
        string_list = evaluation_split.iloc[j, i + 1].split()
        final_list = [word for word in string_list if word not in stop_words]
        evaluation_split.iloc[j, i + 1] = ' '.join(final_list)


development_split['file_name'] = development_split2['file_name']
evaluation_split['file_name'] = evaluation_split2['file_name']

print(development_split.iloc[4,5])
# development_split.to_csv('removed_stopwords_dev.csv', encoding= 'utf-8', index=False)
# evaluation_split.to_csv('removed_stopwords_eval.csv', encoding= 'utf-8', index=False)
#



a = development_split.caption_1.str.split(expand=True).stack().value_counts()
print(a)




# a = development_split.caption_1.str.split(expand=True).stack().value_counts()
# print(a)

# a = development_split.caption_2.str.split(expand=True).stack().value_counts()
# print(a)
#
# a = development_split.caption_3.str.split(expand=True).stack().value_counts()
# print(a)
#
# a = development_split.caption_4.str.split(expand=True).stack().value_counts()
# print(a)
#
# a = development_split.caption_5.str.split(expand=True).stack().value_counts()
# print(a)
#
# a = pd.Series(' '.join(development_split.caption_1).split()).value_counts()[:15]
# print(a)