# check for cuda
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time
import copy
import argparse
import ast
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import lil_matrix

# check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Load the CSV file and extract the Description column
data = pd.read_csv('ANLP-2/train.csv')
corpus = data['Description']
sentence_count= 0
sentence_limit = 40000
sentences_list = []

for sentence in corpus:
    sentences_list.append(sentence)
    sentence_count += 1
    if sentence_count > sentence_limit:
        break

word_counts = {}
text_courpus = []
MinWords = 40000 # min words in a sentence
minSentenceLength = 500 # min words in a sentence
for sentence in sentences_list:
    sentence = sentence.lower() # convert to lower case
    sentence = re.sub(r'[^\w\s]', '', sentence) # remove punctuation
    text_courpus.append(sentence)
    words = sentence.split()
    minSentenceLength = min(minSentenceLength, len(words))
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1



class WordEmbedding:
    def __init__(self, window_size=2, embedding_size=100):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.word2index = {}
        self.index2word = {}
        self.word_counts = {}
        self.word_vectors = None
        self.word_vectors_norm = None
        self.vocab_size = 0
    def buid_Co_occurence_matrix(self, corpus, word_counts):
        self.word2index = {word: i for i, word in enumerate(word_counts.keys())}
        self.index2word = {i: word for word, i in self.word2index.items()}
        self.word_counts = word_counts
        self.corpus = corpus
        vocab = set(word_counts.keys())
        self.vocab_size = len(vocab)
        #Create a n*n matrix where n is the number of words in the vocab
        self.co_occurence_matrix = lil_matrix((self.vocab_size, self.vocab_size), dtype=np.float32)
        for sentence in corpus:
            words = sentence.split()
            for i, word in enumerate(words):
                if word in self.word2index:
                    for j in range(max(i - self.window_size, 0), min(i + self.window_size, len(words))):
                        if i != j and words[j] in self.word2index:
                            self.co_occurence_matrix[self.word2index[word], self.word2index[words[j]]] += 1
    
    def train(self, corpus, word_counts):
        self.buid_Co_occurence_matrix(corpus, word_counts)
        svd = TruncatedSVD(n_components=self.embedding_size)
        svd.fit(self.co_occurence_matrix)
        embeddings = svd.components_
        embeddings = np.transpose(embeddings)
        return embeddings, self.word2index, self.index2word
    


Embedding_Model = WordEmbedding(window_size=3, embedding_size=100)
embeddings, word2index, index2word = Embedding_Model.train(text_courpus, word_counts)


U, s, VT = np.linalg.svd(embeddings, full_matrices = False)


def plot_similar_words(input_word, word2index, embeddings):
    input_word_index = word2index[input_word]
    input_word_embedding = embeddings[input_word_index]
    print('Word vector for the word:', input_word)
    print(input_word_embedding)
    
    word_distances = []
    for word, idx in word2index.items():
        if word == input_word:
            continue
        other_word_embedding = embeddings[idx]
        distance = np.linalg.norm(input_word_embedding - other_word_embedding)
        word_distances.append([word, distance, other_word_embedding])
    
    word_distances.sort(key=lambda x: x[1])
    closest_embeddings = []
    word_labels = []
    print('Top 10 similar words to the input word:', input_word)
    for i in range(10):
        print(word_distances[i][0])
        closest_embeddings.append(word_distances[i][2])
        word_labels.append(word_distances[i][0])
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    np.set_printoptions(suppress=True)
    y = tsne.fit_transform(np.asarray(closest_embeddings))
    x_coords = y[:, 0]
    y_coords = y[:, 1]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, marker='o', color='blue', label='Similar Words')
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Similar Words Visualization')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_similar_words('love', word2index, embeddings)


import json
word_embeddings_tensor = torch.FloatTensor(embeddings)
# Save the embeddings to a file
torch.save(word_embeddings_tensor, 'svd-word-vectors.pt')
# Save word2index and index2word to JSON files
def save_dictionary(dictionary, file_path):
    with open(file_path, 'w') as f:
        json.dump(dictionary, f)

# Load word2index and index2word from JSON files

# Save dictionaries to JSON files
save_dictionary(word2index, 'word2index.json')
save_dictionary(index2word, 'index2word.json')


    
 