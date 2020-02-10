import spacy
import torch
import pandas as pd
from torchtext import data, datasets
from collections import defaultdict
import config

t2i = defaultdict(lambda: len(t2i))
def read_dataset(filename, csv_filename):
    labels = []
    sentences = []
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            if tag == 'media and darama':
                tag = 'media and drama'
            labels.append(t2i[tag])
            sentences.append(words)
    print('writing to csv file.......')
    df = pd.DataFrame({'text' : sentences,
                        'label' : labels})
    #df.to_csv(csv_filename)
    print('successfully write to csv file')
#read_dataset(config.train_data_file, 'train_data.csv')
#read_dataset(config.dev_data_file, 'dev_data.csv')
#read_dataset(config.test_data_file, 'test_data.csv')
print(t2i)
spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_dataset(test_sen=None):

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=150, batch_first = True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train, val, test = data.TabularDataset.splits(
            path='./data_hw1/', train='train_data.csv',
            validation='dev_data.csv', test='test_data.csv', format='csv', skip_header = True, 
            fields=[('text', TEXT), ('label', LABEL)])
    #print(train[0].label)
    TEXT.build_vocab(train, val, test, vectors="glove.6B.100d")
    #LABEL.build_vocab(train, val)
    #train_iter, val_iter = data.Iterator.splits(
            #(train, val), sort_key=lambda x: len(x.text), shuffle = True, 
            #batch_sizes=(32, 256), device=-1)

    train_iter = data.BucketIterator(
            dataset = train, shuffle = True, 
            batch_size=32, device=-1)

    val_iter = data.BucketIterator(
            dataset = val, shuffle = False, 
            batch_size=256, device=-1)
    test_iter = data.BucketIterator(
            dataset = test, shuffle = False, 
            batch_size=256, device=-1)


    #vocab = TEXT.vocab
    word_embeddings = TEXT.vocab.vectors
    #print(word_embeddings)
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    #print(LABEL.vocab)
    #print ("Label Length: " + str(len(LABEL.vocab)))
    vocab_size = len(TEXT.vocab)
    return TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter
