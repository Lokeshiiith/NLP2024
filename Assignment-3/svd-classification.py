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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the embeddings from the saved file
import json
word_embeddings = torch.load('svd-word-vectors.pt')
# Load dictionaries from JSON files
def load_dictionary(file_path):
    with open(file_path, 'r') as f:
        dictionary = json.load(f)
    return dictionary
word2index = load_dictionary('word2index.json')
index2word = load_dictionary('index2word.json')



# -------------------Bilstm---------------
num_sentences = 40001
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings), freeze=True)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])  # Take last timestep's output
        return logits

 
learning_rate = 0.001
# Initialize the model and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(embedding_dim=100, hidden_dim=128, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5
data = pd.read_csv('ANLP-2/train.csv')
class_indices_40000 = data['Class Index'][:num_sentences]



# -------Training the model----------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define lists to store metrics for train set
train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1_scores = []
train_confusion_matrices = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    y_true = []  # True labels
    y_pred = []  # Predicted labels

    for idx in range(len(text_courpus)):
        word_tokens = text_courpus[idx].split()
        # Convert words to indices and then to tensor
        word_indices = [word2index[word] for word in word_tokens if word in word2index]
        word_tensor = torch.LongTensor(word_indices).unsqueeze(0).to(device)
        # Get the class label
        class_label = class_indices_40000[idx]
        class_tensor = torch.LongTensor([class_label]).to(device)

        outputs = model(word_tensor)
        
        loss = criterion(outputs, class_tensor)
        optimizer.zero_grad()  # Clear gradients
        outputs = model(word_tensor)  # Forward pass
        loss = criterion(outputs, class_tensor)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == class_tensor).sum().item()
        total_samples += 1
        
        # Append true labels and predicted labels for metrics calculation
        y_true.append(class_tensor.item())
        y_pred.append(predicted.item())

    # Calculate training metrics after each epoch
    epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    # Calculate precision, recall, f1 score, and confusion matrix
    train_precision = precision_score(y_true, y_pred, average='macro')
    train_precisions.append(train_precision)
    
    train_recall = recall_score(y_true, y_pred, average='macro')
    train_recalls.append(train_recall)
    
    train_f1 = f1_score(y_true, y_pred, average='macro')
    train_f1_scores.append(train_f1)
    
    train_conf_matrix = confusion_matrix(y_true, y_pred)
    train_confusion_matrices.append(train_conf_matrix)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}')




# --------------TEstig the model----------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, criterion, test_corpus, test_class_indices, word2index, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for idx in range(len(test_corpus)):
            word_tokens = test_corpus[idx].split()
            if not word_tokens:
                continue
            # Convert words to indices and then to tensor
            word_indices = [word2index[word] for word in word_tokens if word in word2index]
            if not word_indices:  # Handle case where all words are out-of-vocabulary
                continue
            word_tensor = torch.LongTensor(word_indices).unsqueeze(0).to(device)
            # Get the class label
            class_label = test_class_indices[idx]
            class_tensor = torch.LongTensor([class_label]).to(device)

            outputs = model(word_tensor)
            
            loss = criterion(outputs, class_tensor)

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == class_tensor).sum().item()
            total_samples += 1

            all_predictions.append(predicted.item())
            all_targets.append(class_label)

    epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    # Compute additional metrics
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=1)  # Set zero_division to 1
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=1)  # Set zero_division to 1
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=1)  # Set zero_division to 1
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    print(f'Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

# Example usage:
# Assuming you have 'model', 'criterion', 'word2index', 'device', 'test_corpus', and 'test_class_indices' already defined
evaluate_model(model, criterion, test_corpus, test_class_indices, word2index, device)
































