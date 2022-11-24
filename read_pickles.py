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


#import sklearn
#file_name = "/Users/george/Documents/python/greek/pickles/words_list.p"
#objects = pd.read_pickle(file_name)

reference = [['αυτο', 'ειναι', 'ενα', 'τεστ'], ['αυτο', 'ειναι', 'τεστ']]
candidate = ['αυτο', 'ειναι', 'ενα', 'τεστ', 'για', 'πλακα']
#score = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
#print(score)
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))

full_file_dev = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_development.csv', usecols=[0,1,2,3,4,5])
full_file_eval = pd.read_csv(r'/Users/george/Documents/python/greek/after/clotho_captions_evaluation.csv',  usecols =[0,1,2,3,4,5])
print(full_file_dev.shape)
print(full_file_eval.shape)

full_file = pd.concat([full_file_dev,full_file_eval])
print(full_file.shape)

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

#allsentences = ["Joe waited for the train", "The train was late", "Mary and Samantha took the bus",
               # "I looked for Mary and Samantha at the bus station",
                #"Mary and Samantha arrived at the bus station early but waited until noon for the bus"]

voc = Vocabulary('test')
print(voc)

corpus = ['This is the first sentence.',
          'This is the second.',
          'There is no sentence in this corpus longer than this one.',
          'My dog is named Patrick.']

print(corpus)
for sent in corpus:
  voc.add_sentence(sent)
print('word is appears:')


print('times')
print('Token 4 corresponds to token:', voc.to_word(11))
print('Token "this" corresponds to index:', voc.to_index('first'))

docs = full_file
print(docs.shape)
print(full_file_dev.iloc[0,2])
print(full_file.iloc[0,1])
#print(docs.head)
allsentences = docs.iloc[3,:]
#print(allsentences)
#generate_bow(allsentences)
print('a')
#print(docs.iloc[1:10,2])
#print(full_file.iloc[1:10,2])
#docs.to_csv('the_extraction.csv')

the_list = []
with open('/Users/george/Documents/python/greek/the_extraction.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        the_list.append((row['caption_1'], row['caption_2'], row['caption_3'], row['caption_4'], row['caption_5']))

print(type(the_list))
word = 'ενα'
print(the_list[1])
tokenized = nltk.word_tokenize(str(the_list))

if word in tokenized:
    print('Yes, found at list')
else:
    print('not found in list')
test2 = full_file_eval.iloc[3, 4]
protasi = 'ο ρυθμος του κτιριου ηχει και κανει θορυβους στο υπολοιπο κτιριο'

nlp = spacy.load("el_core_news_md")
sentence = "Δεν εχει ποια καποια αναγκη προσεγγισης των δανειων της υπολογιστικης"
doc = nlp(test2)
empty_list = []
for token in doc:
    empty_list.append(token.lemma_)
final_string = ' '.join(map(str,empty_list))
print(final_string)
print(full_file_eval.iloc[3,4])

stemmed_word = stemmer.stem_word('εργαζόμενος', 'VBG')
stemmed_word =stemmed_word.lower()
print(stemmed_word)
stemmed = full_file_eval
for i in range(5):
    for j in range(1045):
        list1 = []
        lst = nltk.word_tokenize(full_file_eval.iloc[j, i+1])
        stemmed.iloc[j, i+1] = ''
        for k in lst:
            k1 = stemmer.stem_word(k, 'VBG')
            k1 = k1.lower()
            stemmed.iloc[j, i + 1] = stemmed.iloc[j, i + 1] + k1 + ' '
            #stemmed.iloc[j, i + 1].append(k1)
            #stemmed.iloc[j, i+1 ] =' '.join(k1)
            #stemmed.iloc[j, i]=stemmer.stem_word(full_file_eval.iloc[j, i], 'VBG')

uniqueWords = list(set(" ".join(full_file_eval).lower().split(" ")))
count = len(uniqueWords)
print(count)

uniqueWords2 = list(set(" ".join(stemmed).lower().split(" ")))
count2 = len(uniqueWords2)
print(count2)
print(stemmed.shape)
#stemmed.to_csv('stemmed.csv', encoding= 'utf-8-sig')
for i in range(5):
    for j in range(full_file.shape[0]-1):
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
voc2 = Vocabulary('test2')
for i in range(5):
    for j in range(full_file_eval.shape[0] - 1): #shape: (2893, 6)
        prot = full_file_eval.iloc[j, i+1]
        voc2.add_sentence(prot)

#print(voc2.num_words)
# eval: -3022- 3019 words
# dev: -9883- 7163 words
#full_file_dev.to_csv('tlbmnsq.csv', encoding='utf-8-sig')
#full_file.to_csv('full_file.csv', encoding ='utf-8-sig')
nlp = spacy.load("el_core_news_lg")
#sentence2 = nlp('το άγνωστο τρίξιμο εμφανίζεται πολλές φορές ενώ τα μπουφάν των ανδρών μιλούν στο παρασκήνιο')
#for word in sentence2:
#    print(word.text,  word.lemma_)
#### sampling
#sampling = full_file.sample(n = 400)
#print(sampling.shape)

########### test in english ##############
#full_file_dev = pd.read_csv(r'/Users/george/Documents/python/greek/clotho_captions_development_2_1_english.csv', usecols=[0,1,2,3,4,5])
#full_file_eval = pd.read_csv(r'/Users/george/Documents/python/greek/clotho_captions_evaluation_2_1_english.csv',  usecols =[0,1,2,3,4,5])
print(full_file_dev.shape)
print(full_file_eval.shape)

full_file = pd.concat([full_file_dev,full_file_eval])
print(full_file.shape)
##########################################
minimum = 300
state_number = 0
print('#############################')
listofwords = []
eval_list = []
for lam in range(1):
    rnd_st = lam + 1100
    tyxaio = numpy.random.RandomState(seed=rnd_st)
    sampling = full_file.sample(n=200, random_state=tyxaio) #minimum=152 words in randum_state=1


    vocsample = Vocabulary('test3')
    for i in range(5):
        for j in range(sampling.shape[0] - 1): #shape: (n, 6)
            prot2 = sampling.iloc[j, i+1]
            vocsample.add_sentence(prot2)
    for m in range(vocsample.num_words):
        eval_list.append(vocsample.index2word[m])
    listofwords.append(vocsample.num_words)
    rest_part = full_file.drop(sampling.index)

    word_list = []
    #with open('/Users/george/Documents/python/greek/full_file.csv') as csvfile:
    #    reader = csv.DictReader(csvfile)
    #    for row in reader:
    #        word_list.append((row['caption_1'], row['caption_2'], row['caption_3'], row['caption_4'], row['caption_5']))
    ###################
    full_vocab = Vocabulary('test4')
    for i in range(5):
        for j in range(full_file.shape[0] - 1): #shape: (poly, 6)
            protasi = full_file.iloc[j, i+1]
            full_vocab.add_sentence(protasi)

    for m in range(full_vocab.num_words):
        word_list.append(full_vocab.index2word[m])
    #print('total words:')
    #print (len(word_list))
    #print('words in supposed eval split:')
    #print(vocsample.num_words)
    ################## rest ###################
    rest_vocab = Vocabulary('test5')
    for i in range(5):
        for j in range(rest_part.shape[0] - 1): #shape: (poly, 6)
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
print(state_number)