import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import numpy
import re
import csv
import spacy
from greek_stemmer import stemmer
import unicodedata

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()


class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));

    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))

##########################################################
full_file_dev = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_development.csv', usecols=[0,1,2,3,4,5], encoding='utf-8-sig')
full_file_eval = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_evaluation.csv',  usecols =[0,1,2,3,4,5], encoding='utf-8-sig')
print(full_file_dev.shape)
print(full_file_eval.shape)

full_file = pd.concat([full_file_dev,full_file_eval])
print(full_file.shape)
#### NEW
#full_file = pd.read_csv(r'/Users/george/Documents/python/greek/small_dataset.csv', usecols=[0,1,2,3,4,5], encoding='utf-8-sig') ## added
#######
##### remove accents, commas, full stops and turn lowercase ###########
for i in range(5):
    for j in range(full_file.shape[0]):
        full_file.iloc[j, i+1] = strip_accents_and_lowercase(full_file.iloc[j,i+1])
        full_file.iloc[j, i + 1] = re.sub(r'[.,"\'-?:!;]', '', full_file.iloc[j, i+1])

for i in range(5):
    for j in range(full_file_eval.shape[0]-1):
        full_file_eval.iloc[j, i+1] = strip_accents_and_lowercase(full_file_eval.iloc[j,i+1])
        full_file_eval.iloc[j, i + 1] = re.sub(r'[.,"\'-?:!;]', '', full_file_eval.iloc[j, i+1])

for i in range(5):
    for j in range(full_file_dev.shape[0]-1):
        full_file_dev.iloc[j, i+1] = strip_accents_and_lowercase(full_file_dev.iloc[j,i+1])
        full_file_dev.iloc[j, i + 1] = re.sub(r'[.,"\'-?:!;]', '', full_file_dev.iloc[j, i + 1])
##############
####### create splits
print('####################################')
# nlp = spacy.load("el_core_news_md")
#
# empty_list = []
maximum = 15
print(full_file.shape)
for i in range(5):
    for j in range(full_file.shape[0]):
        sum = 0
        sentence = full_file.iloc[j, i+1]
        for word in sentence.split():
            sum = sum +1
            if sum > maximum:
                maximum = sum
print('maximum is:', maximum)

#creating lemmatized full file
# for i in range(5):
#     #for j in range(full_file.shape[0]-1):
#     for j in range(full_file.shape[0]-1):
#         sentence = full_file.iloc[j,i+1]
#         doc = nlp(sentence)
#         empty_list = []
#         for token in doc:
#             empty_list.append(token.lemma_)
#             full_file.iloc[j,i+1] = ' '.join(map(str,empty_list))


# print('#######')
# sentence = 'βιβλίου βιβλίων βιβλιου βιβλιων αναστεναξε καθως γυριζε τις σελιδες του βιβλιου σταματωντας να σαρωσει τις πληροφοριες'
# doc2 = nlp(sentence)
# next_list = []
# next_list2 = []
# for token in doc2:
#     next_list.append(token.lemma_)
#
# print (next_list)
# print('##########')

#full_file.to_csv('lemmatized_full.csv', encoding='utf-8')

#full_file_dev = pd.read_csv(r'/Users/george/Documents/python/greek/lemmatized_full.csv', usecols=[0,1,2,3,4,5], encoding='utf-8-sig')

minimum = 600
#state_number = 0
print('#############################')

for lam in range(1):
    print('loop:' + str(lam))
    rnd_st = 3593 #lam + 4000
    tyxaio = numpy.random.RandomState(seed= rnd_st)
    sampling = full_file.sample(frac=0.2, random_state=tyxaio) #minimum=152 words in randum_state=1
    #sampling = pd.read_csv(r'/Users/george/Documents/python/greek/small_dataset_eval.csv', encoding = 'utf-8-sig', usecols=[0,1,2,3,4,5])  #NEW

    #min = 136 in rand_state = numpy.random.RandomState(seed=1303) and (seed=1582)
    #min = 128 in rand_state = numpy.random.RandomState(seed = 2104)
    #for frac=0.1, min=299 for rnd_st=2625// min=293 for rnd_st=3593//

    eval_list = []
    vocsample = Vocabulary('test3')
    for i in range(5):
        for j in range(sampling.shape[0] - 1): #shape: (n, 6)
            prot2 = sampling.iloc[j, i+1]
            vocsample.add_sentence(prot2)
    for m in range(vocsample.num_words):
        eval_list.append(vocsample.index2word[m])
    #print(eval_list)####
    print(vocsample.num_words)
    #print(len(eval_list))


    rest_part = full_file.drop(sampling.index)
    rest_part = pd.read_csv(r'/Users/george/Documents/python/greek/small_dataset_dev.csv', encoding = 'utf-8-sig', usecols=[0,1,2,3,4,5]) #NEW
    #print('rest part shape is:', rest_part.shape) #new
    #rest_part.drop('Unnamed: 0', inplace=True, axis=1)
    #rest_part = rest_part.loc[:, ~rest_part.columns.str.contains('^Unnamed')] # NEW
    rest_vocab = Vocabulary('test5')
    for i in range(5):
        for j in range(rest_part.shape[0]): #shape: (all - n, 6)
            protasi = rest_part.iloc[j, i+1]
            rest_vocab.add_sentence(protasi)
    word_list_rest = []
    for m in range(rest_vocab.num_words):
        word_list_rest.append(rest_vocab.index2word[m])
    #print('words in supposed dev split:')
    #print (len(word_list_rest))
    ####################

    checker = 0
    not_checker = 0

    for i in range(len(eval_list)):
        if eval_list[i] in word_list_rest:
            checker = checker + 1
        else:
            not_checker= not_checker + 1

    if not_checker < minimum:
        minimum = not_checker
        state_number = lam

    #print('words included')
    #print(checker)
    #print('not included:')
    #print(not_checker)

    #sampling.to_csv('an_example_cut.csv', encoding='utf-8')

print('minimum:')
print(minimum)
print('state number:')
#print(state_number)

print('number of words:'+ str(vocsample.num_words))

#for idx in range(vocsample.num_words):
    #print(vocsample.index2word[idx])
    #print('this word appears' + str(vocsample.word2count)+'times')

    #vocsample.to_word(idx)
print(type(vocsample.word2count))


'''
sent_list = []
for i in range(5):
    for j in range(sampling.shape[0] - 1):  # shape: (n, 6)
        sent_list.append(sampling.iloc[j,i+1])

generate_bow(sent_list)
'''
for i in range(5):
    for j in range (sampling.shape[0]): #vazoume -1 'h oxi?
        docu = sampling.iloc[j, i+1]
        list_remove = []
        for each_word in docu.split():
            if each_word not in word_list_rest:
                list_remove.append(each_word)
            #print(list_remove)
        for index in range(len(list_remove)):
            s = list_remove[index]
            sampling.iloc[j, i+1] = sampling.iloc[j,i+1].replace(s, '')
#print(docu)
print(rest_part.shape)
print(sampling.shape)
print('old words:')

#sampling.to_csv('removed_extra_words_cut_EVAL_10.csv', encoding= 'utf-8')
#rest_part.to_csv('removed_extra_words_cut_DEV_90.csv', encoding= 'utf-8')

##### New

#sampling.to_csv('small_removed_extra_words_cut_EVAL_20.csv', encoding= 'utf-8', index=False)
#rest_part.to_csv('small_removed_extra_words_cut_DEV_80.csv', encoding= 'utf-8', index=False)


final_list = []
voc_final = Vocabulary('test5')
for i in range(5):
    for j in range(full_file.shape[0]):  # shape: (n, 6)
        prot3 = full_file.iloc[j, i + 1]
        voc_final.add_sentence(prot3)
for m in range(voc_final.num_words):
    final_list.append(voc_final.index2word[m])

print('final list length:', len(final_list))
# print(eval_list)####

