import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import re
import random
import torchtext
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                    dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)  
        self.init_weights()

    def forward(self, src, hidden):
                
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, 
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

#helper functions 

def tokenize(text):
    url_re = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    mention_re = "@\w+"
    hastag_re = "#[a-z0-9_]+"
    text = text.lower()
    text = re.sub(url_re, '<url> ', text)
    text = re.sub(hastag_re, '<hashtag> ', text)
    text = re.sub(mention_re, '<mention> ', text)
    tokens = re.findall(r'\b\w+|[^\s\w<>]+|<\w+>', text)
    return tokens

def test_train_val_split(text ,t_n, v_n):
    sentences = re.split(r'(?<=[.!?]) +', text)
    test_sen = random.sample(sentences, t_n)
    train_sen = [sentence for sentence in sentences if sentence not in test_sen]
    val_sen = random.sample(train_sen, v_n)
    train_sen = [sentence for sentence in sentences if sentence not in test_sen or val_sen]
    return [test_sen, train_sen, val_sen]

def get_word_count(tokens):
    word_count = {}
    
    for word in tokens:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def add_unk(word_count, tokens, thr):
    for i in range(len(tokens)):
        try:
            if(word_count[tokens[i]]<thr):
                tokens[i] = "<unk>"
        except KeyError:
            tokens[i] = "<unk>"
    return tokens

def get_data(tok, vocab, batch_size):
    data = []                                     
    tokens = tok.append('<eos>')             
    tokens = [vocab[token] for token in tok] 
    data.extend(tokens)
    num_batches = len(data) // batch_size  
    if num_batches == 0:
         data_len = len(data)
         num_padding = batch_size - data_len
         data.extend([vocab['<pad>']] * num_padding)
         num_batches = 1                           
    data = torch.LongTensor(data)                                 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data

def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    
    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):  # The last batch can't be a src
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)               

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def train_model(n_epochs=30, seq_len=25, saved=False, saved_model=None ):
    clip = 0.25

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    if saved:
        model.load_state_dict(torch.load(saved_model,  map_location=device))
        test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {math.exp(test_loss):.3f}')
    else:
        print("No saved model found")
        print("Training model")
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = train(model, train_data, optimizer, criterion, 
                        batch_size, seq_len, clip, device)
            valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                        seq_len, device)
            
            lr_scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), saved_model)
            
            print(f'\tEpoch : {epoch+1}')
            print(f'\t\t\tTrain Perplexity: {math.exp(train_loss):.3f}')
            print(f'\t\t\tValid Perplexity: {math.exp(valid_loss):.3f}\n')

def get_perplexity_of_sen(sen, wc, saved_model):
    toks = tokenize(sen)
    toks = add_unk(wc, toks, 5)
    toks = [vocab[token] for token in toks] 
    data = torch.LongTensor([toks])
    model.load_state_dict(torch.load(saved_model,  map_location=device))
    sen_loss = evaluate(model, data, criterion, 1, data.shape[1]-1, device)
    sen_perplexity = math.exp(sen_loss)
    return sen_perplexity

def process_data(test_train_val_sen, wc):
    test_sen, train_sen, val_sen = test_train_val_sen
    for i in range(len(test_sen)):
        test_sen[i] = test_sen[i] + "<eos>"
    for i in range(len(train_sen)):
        train_sen[i] = train_sen[i] + "<eos>"
    for i in range(len(val_sen)):
        val_sen[i] = val_sen[i] + "<eos>"
        
    train_d = tokenize("".join(train_sen))
    test_d = tokenize("".join(test_sen))
    val_d = tokenize("".join(val_sen))

    train_d = add_unk(wc, train_d, 5)
    test_d = add_unk(wc, test_d, 5)
    val_d = add_unk(wc, val_d, 5)

    return [test_d, train_d, val_d]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        saved = False
    elif len(sys.argv) == 3:
        saved = True
        saved_model = sys.argv[2]
    else:
        print("Correct usage format is python3  neural_language_model.py <corpus> <trained_model_path>(optional)")
        exit()
    
    corpus = sys.argv[1]


    with open(corpus, "r") as file:
        text = file.read()

    test_sen, train_sen, val_sen = test_train_val_split(text, 1000, 1000)

    train_tokens = tokenize("".join(train_sen))
    wc = get_word_count(train_tokens)
    train_it = []
    for sen in train_sen:
        updated_sen = add_unk(wc, tokenize(sen), 5)
        train_it.append(updated_sen)


    #getting the vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(train_it) 
    vocab.insert_token('<pad>', 0)          
    vocab.set_default_index(vocab['<unk>'])  

    test_d, train_d, val_d = process_data([test_sen, train_sen, val_sen], wc)



    #hyperparameters
    batch_size = 128
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout_rate = 0.2             
    lr = 0.001


    train_data = get_data(train_d, vocab, batch_size)
    valid_data = get_data(val_d, vocab, batch_size)
    test_data = get_data(test_d, vocab, batch_size)
    model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if not saved:
        #training the model
        train_model(saved=saved, saved_model="lstm_ulysses_model.pt")
        saved_model = "lstm_ulysses_model.pt"
    
    # with open("Ulysses_LM6_train-perplexity.txt", "w") as f:
    #     w_sum = 0
    #     for sentence in train_sen:
    #         perplexity = per_in_sen = get_perplexity_of_sen(sentence, wc, saved_model)
    #         w_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_w_train_perplexity = w_sum/len(train_sen)
    #     line = "Avg perplexity: " + str(avg_w_train_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()

    # with open("Ulysses_LM6_test-perplexity.txt", "w") as f:
    #     w_sum = 0
    #     for sentence in test_sen:
    #         perplexity = per_in_sen = get_perplexity_of_sen(sentence, wc, saved_model)
    #         w_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_w_test_perplexity = w_sum/len(test_sen)
    #     line = "Avg perplexity: " + str(avg_w_test_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()

    
    quit = False
    while not quit:
        in_sen = input("Enter a sentence: ")
        per_in_sen = get_perplexity_of_sen(in_sen, wc, saved_model)
        print("Perplexity: ",per_in_sen)
        c = input("\nContinue(y/n)")
        if c != 'y':
            quit = True
