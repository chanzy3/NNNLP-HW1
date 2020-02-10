import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict
from collections import Counter
import random
from torch.utils import data
import config
import time
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader
import os
import spacy


topics = ['sports and recreation', 'social sciences and society', 'media and drama', 'warfare', \
          'engineering and technology', 'language and literature', 'history', 'mathematics', \
          'philosophy and religion', 'art and architecture', 'video games', 'miscellaneous', \
          'music', 'natural sciences', 'agriculture, food and drink', 'geography and places']


spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

#input directory should only contain data files
def build_vocab(directory):
    #print(os.listdir(directory))
    unique_words = set()
    file_lst = os.listdir(directory)
    for filename in file_lst:
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename)) as f:
                for line in f:
                    _, context = line.lower().strip().split(' ||| ')
                    words = tokenizer(context)
                    for word in words:
                        if word not in unique_words:
                            unique_words.add(word)
    print("We find", len(unique_words), " unique words")

    return unique_words


def create_word_embedding(unique_words):
    glove_embedding = {}
    with open("./glove.6B.100d.txt") as f:

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_embedding[word] = coefs

    #find the words that exist both in glove and dictionaries and assign rest words to unknown
    #common_words = glove_embedding.keys() & unique_words

    embedding_matrix = np.zeros((len(unique_words), config.EMB_DIM))

    #reverse seats for unknown word and initialize it with random word in embedding
    #embedding_matrix[0] = random.choice(list(glove_embedding.values()))
    w2i = {}
    for i, word in enumerate(unique_words):
        if word in glove_embedding.keys():
            embedding_matrix[i] = glove_embedding[word]
        else:
            #If you can't find words in glove_embedding randomize it with random values in glove embedding
            embedding_matrix[i] = random.choice(list(glove_embedding.values()))
        w2i[word] = i
    # embedding_matrix['unk'] = embeddings_index.get('unk')
    #print(embedding_matrix)
    return w2i, embedding_matrix

def data_loader(filename, w2i):
    X = []
    y = []
    with open(filename) as f:
        for line in f:
            topic, context = line.lower().strip().split(' ||| ')
            if topic == 'media and darama':
                topic = 'media and drama'

            y.append(topics.index(topic))

            words = tokenizer(context)

            #if you can find word in vocab, return index, else return index for unknown
            X.append([w2i[word] if word in w2i.keys() else w2i["UNK"] for word in words])

    return X, y

class MyDataSet(Dataset):
    def __init__(self,X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self,i):
        return self.X[i], self.Y[i]
    def __len__(self):
        return len(self.X)

def collate_sequence(seq_list):
    inputs,targets = zip(*seq_list)

    inputs = [torch.LongTensor(input) for input in inputs]
    targets = torch.LongTensor(targets)

    inputs = pad_sequence(inputs, batch_first = True)

    return inputs,targets


