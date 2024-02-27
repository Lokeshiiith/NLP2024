import re
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import os
import json

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



# This will provide each sentence with paddings <unk>
def process_dataset(dataset_file, p=2, s=3):
    sentences_list = []
    pos_list = []

    with open(dataset_file, 'r', encoding='utf-8') as f:
        sentence_tokens = []
        pos_tags = []

        for line in f:
            line = line.strip()

            if line.startswith('#'):
                sentence_tokens = []
                pos_tags = []
                continue
            elif line == '':
                # Append padding to the end of the sentence
                padded_sentence = ' '.join(['<PAD>'] * p) + ' ' + ' '.join(sentence_tokens) + ' ' + ' '.join(['<PAD>'] * s)
                padded_pos = ' '.join(['<UNK>'] * p + pos_tags + ['<UNK>'] * s)
                sentences_list.append(padded_sentence)
                pos_list.append(padded_pos)
                continue
            else:
                # New sentence begins
                token_attrs = line.split('\t')
                word_form = token_attrs[1]  # Word form of the token
                pos_tag = token_attrs[3]    # POS tag of the token
                sentence_tokens.append(word_form)
                pos_tags.append(pos_tag)
    return sentences_list, pos_list

# Get indices for each word and tag
def get_indices(sentences_list, pos_list, word_to_index, tag_to_index, max_sentence_length, word_count):
    # Process each sentence to tokenize the data
    # Get indices for each word and tag
    path = "."
    train_dataset = path + "/conllu/train.conllu"
    test_dataset = path + "/conllu/test.conllu"
    val_dataset = path + "/conllu/val.conllu"
    train_sentece_list, train_pos_list = process_dataset(train_dataset, p= 2, s= 3)
    test_sentece_list, test_pos_list = process_dataset(test_dataset, p= 2, s= 3)
    val_sentece_list, val_pos_list = process_dataset(val_dataset, p= 2, s= 3)


    sentences_list = train_sentece_list + test_sentece_list + val_sentece_list
    pos_list = train_pos_list + test_pos_list + val_pos_list
    for sentence_str, tag_str in zip(sentences_list, pos_list):
        # Tokenize the sentence into individual tokens
        tokens = sentence_str.split(' ')
        tags = tag_str.split(' ')
            # Word to index
        for word, tag in zip(tokens, tags):
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
            word_count[word] = word_count.get(word, 0) + 1
        # Tag to index
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)
        max_sentence_length = max(max_sentence_length, len(tokens))
    return word_to_index, tag_to_index, max_sentence_length, word_count





 

def PrepareEmbedding(sentence_dataset, pos_dataset, word_to_index, tag_to_index, max_sentence_length, word_count):
    token_embeddings = []
    labels_embedding = []
    for sentence_str, tag_str in zip(sentence_dataset, pos_dataset):
        # Tokenize the sentence into individual tokens
        tokens = sentence_str.split(' ')
        tags = tag_str.split(' ')
        one_sentence_token_embedding = []
        one_sentence_pos_embedding = []
        # Word to index
        for word, tag in zip(tokens, tags):
            if word in word_to_index:
                if word_count[word] < 2:
                    word_cur_idx = word_to_index['<UNK>']
                else:
                    word_cur_idx = word_to_index[word]
            else:
                word_cur_idx = word_to_index['<UNK>']
            # Tag to index
            if tag in tag_to_index:
                tag_cur_idx = tag_to_index[tag]
            else:
                tag_cur_idx = tag_to_index['<UNK>']
            one_sentence_token_embedding.append(word_cur_idx)
            one_sentence_pos_embedding.append(tag_cur_idx)
        # Pad sequences using PyTorch's pad_sequence function
        token_embeddings.append(one_sentence_token_embedding)
        labels_embedding.append(one_sentence_pos_embedding)
    return token_embeddings, labels_embedding


# train_sentence_embeddings, train_pos_embeddings = PrepareEmbedding(train_sentece_list, train_pos_list, word_to_index, tag_to_index, max_sentence_length, word_count)
# test_sentence_embeddings, test_pos_embeddings = PrepareEmbedding(test_sentece_list, test_pos_list, word_to_index, tag_to_index, max_sentence_length, word_count)
# val_sentence_embeddings, val_pos_embeddings = PrepareEmbedding(val_sentece_list, val_pos_list, word_to_index, tag_to_index, max_sentence_length, word_count)



# --------------------feed forward neural network--------------------
# --------------------feed forward neural network--------------------
# --------------------feed forward neural network--------------------
# --------------------feed forward neural network--------------------
# Step 1: Define the Model
class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, p, s):
        super(FFNN, self).__init__()
        # Calculate the actual input size considering embedding dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear((p + s + 1) *embedding_dim , hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input tensor
        first = self.embedding(x)
        first = first.view(-1)
        out = self.fc1(first)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# # Define the number of steps after which to print the loss and accuracy
# print_interval = 10
# # Step 2: Define Loss Function
# criterion = nn.CrossEntropyLoss()
# # Step 3: Instantiate Model
# vocab_size = len(word_to_index)
# embedding_dim = 100  # Example dimension, adjust as needed
# hidden_size = 128    # Example size, adjust as needed
# output_size = len(tag_to_index)
# p = 2
# s = 3
# model = FFNN(vocab_size, embedding_dim, hidden_size, output_size, p, s)
# optimizer = optim.Adam(model.parameters(), lr=0.001) # Example optimizer, adjust as needed

def train_model(model, criterion, optimizer, train_embeddings, train_pos_embeddings, print_interval=10):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Iterate over the training dataset
    for token_indices, pos_indices in zip(train_embeddings, train_pos_embeddings):
        # Create sliding window of size 6 and convert to tensors
        for i in range(p, len(token_indices) - s):
            window_tokens = token_indices[i-p:i+s+1]
            window_tokens_tensor = torch.LongTensor(window_tokens)
            pos_tag = pos_indices[i]
            # creating one hot encoding for the pos tag
            # length should be the number of tags
            pos_tag_tensor = torch.zeros(len(tag_to_index))
            pos_tag_tensor[pos_tag] = 1
            optimizer.zero_grad()
            outputs = model(window_tokens_tensor)  # Forward pass
            # Calculate loss
            loss = criterion(outputs, pos_tag_tensor)  # Compare outputs with true labels
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(train_embeddings)

def evaluate_model(model, criterion, val_embeddings, val_pos_embeddings):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    # Iterate over the validation dataset
    with torch.no_grad():
        for token_indices, pos_indices in zip(val_embeddings, val_pos_embeddings):
            # Create sliding window of size 6 and convert to tensors
            for i in range(p, len(token_indices) - s):
                window_tokens = token_indices[i-p:i+s+1]
                window_tokens_tensor = torch.LongTensor(window_tokens)
                pos_tag = pos_indices[i]
                # creating one hot encoding for the pos tag
                # length should be the number of tags
                pos_tag_tensor = torch.zeros(len(tag_to_index))
                pos_tag_tensor[pos_tag] = 1
                outputs = model(window_tokens_tensor)  # Forward pass
                # Calculate loss
                loss = criterion(outputs, pos_tag_tensor)  # Compare outputs with true labels
                running_loss += loss.item()

    return running_loss / len(val_embeddings)

def test_model(model, criterion, test_embeddings, test_pos_embeddings):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    # Iterate over the test dataset
    with torch.no_grad():
        for token_indices, pos_indices in zip(test_embeddings, test_pos_embeddings):
            # Create sliding window of size 6 and convert to tensors
            for i in range(p, len(token_indices) - s):
                window_tokens = token_indices[i-p:i+s+1]
                window_tokens_tensor = torch.LongTensor(window_tokens)
                pos_tag = pos_indices[i]
                # creating one hot encoding for the pos tag
                # length should be the number of tags
                pos_tag_tensor = torch.zeros(len(tag_to_index))
                pos_tag_tensor[pos_tag] = 1
                outputs = model(window_tokens_tensor)  # Forward pass
                # Calculate loss
                loss = criterion(outputs, pos_tag_tensor)  # Compare outputs with true labels
                running_loss += loss.item()

    return running_loss / len(test_embeddings)

# # Number of epochs
# num_epochs = 10
# for epoch in range(num_epochs):
#     # Training phase
#     train_loss = train_model(model, criterion, optimizer, train_sentence_embeddings, train_pos_embeddings, print_interval)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

#     # Validation phase
#     val_loss = evaluate_model(model, criterion, val_sentence_embeddings, val_pos_embeddings)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

# Testing phase
# test_loss = test_model(model, criterion, test_sentence_embeddings, test_pos_embeddings)
# print(f"Test Loss: {test_loss:.4f}")


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluateFFNN(model, sentences, word_to_index, tag_to_index, device):
    sentence_token = sentences.split(' ')
    embedded_sentence = [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence_token]
    embedded_sentence = torch.tensor(embedded_sentence, dtype=torch.long)
    token = []
    tag = []
    for i in range(2, len(embedded_sentence) - 3):
        input = embedded_sentence[i-2:i+3 + 1]
        output = model(input)
        token.append(sentence_token[i])
        tag.append(list(tag_to_index.keys())[list(tag_to_index.values()).index(torch.argmax(output).item())])

    for i in range(len(token)):
        print(f"{token[i]}: {tag[i]}")

# --------------------LSTM----------------------------------------------------
# --------------------LSTM---------------------------------------------------
# --------------------LSTM---------------------------------------------------
# --------------------LSTM-----------------------------------------------------
# --------------------LSTM------------------------------------------------------

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_sentence):
        embeds = self.word_embeddings(input_sentence)
        lstm_out, _ = self.lstm(embeds.view(len(input_sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(input_sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    


# def trainModel(model):
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     loss_function = nn.CrossEntropyLoss()
#     for epoch in range(10):
#         model.zero_grad()
#         # ?PrepareEmbedding
#         for sentence, tags in zip(train_sentence_embeddings, train_pos_embeddings):
#             sentence_in = torch.LongTensor(sentence).to(device)
#             targets = torch.LongTensor(tags).to(device)
#             tag_scores = model(sentence_in)
#             loss = loss_function(tag_scores, targets)
#             loss.backward()
#             optimizer.step()
# def testModel(model, test_sentence_embeddings, test_pos_embeddings):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     with torch.no_grad():  # No need to track gradients during testing
#         for sentence, tags in zip(test_sentence_embeddings, test_pos_embeddings):
#             sentence_in = torch.LongTensor(sentence).to(device)
#             targets = torch.LongTensor(tags).to(device)
#             tag_scores = model(sentence_in)
#             _, predicted = torch.max(tag_scores, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy * 100:.2f}%")
#     return accuracy
# model = LSTMTagger(embedding_dim, hidden_dim, len(word_to_index), len(tag_to_index)).to(device)
# trainModel(model)
# testModel(model, test_sentence_embeddings, test_pos_embeddings)



def evaluateLstm(sentence, model, word_to_index, tag_to_index, index_to_tag):
    sentence = re.sub('[^ A-Za-z0-9]+', '', sentence).split()
    tokenized_sent = []
    for word in sentence:
        if word in word_to_index:
            tokenized_sent.append(word_to_index[word])
        else:
            tokenized_sent.append(word_to_index['<UNK>'])

    inputs = torch.tensor(tokenized_sent, dtype=torch.long, device=device)
    output = model(inputs)
    for i in range(len(sentence)):
        print(sentence[i]+"    "+index_to_tag[torch.argmax(output[i]).item()])














if __name__ == "__main__":
    while True:
        import ipdb
        print("Starting")
        # ipdb.set_trace()
        if len(sys.argv) != 2 or sys.argv[1] not in ["-f", "-r"]:
            print("Usage: python pos_tagger.py [-f | -r]")
            sys.exit(1)
        p = 2
        s = 3
        word_to_index = {}
        tag_to_index = {}
        max_sentence_length = 0
        word_count = {}
        word_to_index['<PAD>'] = 0
        tag_to_index['<UNK>'] = 0
        if os.path.exists("word_to_index.json") and os.path.exists("tag_to_index.json") and os.path.exists("word_count.json"):
            with open("word_to_index.json", "r") as f:
                word_to_index = json.load(f)
            with open("tag_to_index.json", "r") as f:
                tag_to_index = json.load(f)
            with open("word_count.json", "r") as f:
                word_count = json.load(f)
        else:
            # Split the data into sentences
            word_to_index = {'<UNK>': 0}
            tag_to_index = {'<UNK>': 0}
            word_count = {'<UNK>': 1}
            max_sentence_length = 0
            # get all indices
            word_to_index, tag_to_index, max_sentence_length, word_count = get_indices(word_to_index, tag_to_index, max_sentence_length, word_count)
        
        index_to_tag = {v: k for k, v in tag_to_index.items()}
        sentence = input("Enter a sentence or type exit to quit: ")
        if sentence == "exit":
            break
        if sys.argv[1] == "-f":
            # load the model
            if not os.path.exists("FFNNModel.pth"):
                print("Model not found. Please train the model first.")
                exit(1)
            # Step 3: Instantiate Model
            vocab_size = len(word_to_index)
            embedding_dim = 100  # Example dimension, adjust as needed
            hidden_size = 64  # Example size, adjust as needed
            output_size = len(tag_to_index)
            model = FFNN(vocab_size, embedding_dim, hidden_size, output_size, p, s)
            model_param = torch.load('FFNNmodel.pth') 
            model.load_state_dict(model_param)
            padded_sentence = ' '.join(['<PAD>'] * p) + ' ' + ''.join(sentence) + ' ' + ' '.join(['<PAD>'] * s)
            evaluateFFNN(model, padded_sentence, word_to_index, tag_to_index, device)
        else :
            # load the model
            if not os.path.exists("lstmModel.pth"):
                print("Model not found. Please train the model first.")
                exit(1)
            vocab_size = len(word_to_index)
            tagset_size = len(tag_to_index)
            embedding_dim = 100
            hidden_dim = 128  
            model_param = torch.load("lstmModel.pth", map_location=torch.device('cpu'))
            model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size)
            model.load_state_dict(model_param)
            padded_sentence = ' '.join(['<PAD>'] * p) + ' ' + ''.join(sentence) + ' ' + ' '.join(['<PAD>'] * s)
            # test the model
            evaluateLstm(sentence, model, word_to_index, tag_to_index, index_to_tag)






    #  main
