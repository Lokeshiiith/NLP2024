

# Files Contained in the Directory

This directory contains the following files:

- 12 text files:
    - 8 files: n-gram model with smoothing techniques
    - 4 files: neural language model
- 2 `.py` files:
    - `language_model.py`
    - `neural_language_model.py`
- `language_model_report.pdf`
- `README.md`
- path to 2 model `.pt`/`.pth` files (one for each corpus) is mentioned in the README:
    - `lstm_pride_model.pt`
    - `lstm_ulysses_model.pt`

## You can find the file at the following one drive location:

- `lstm_pride_model.pt`
[File location](https://drive.google.com/file/d/1SYeyM4OsOLV5aAwHZc522HuaDeTeQj2f/view?usp=sharing)

- `lstm_ulysses_model.pt`
[File location](https://drive.google.com/file/d/1miOAiKoxdfzxizumZImnbJuBKO474Gq_/view?usp=sharing)

# How to Execute the Program

## N-Gram Model

File: `language_model.py`

Execution: `python3 language_model.py <smoothing type> <path to corpus>`

- Smoothing type can be `k` for Kneser-Ney or `w` for Witten-Bell.
- On running the file, the expected output is a prompt, which asks for a sentence and provides the probability of that sentence using the two smoothing mechanisms.

## LSTM Neural Network Model

File: `neural_language_model.py`

Execution: `python3 language_model.py <path to corpus> <trained_model>`

- Here `<trained_model>` argument is optional. If this argument is not provided, the model will be trained with 30 epochs on the given corpus.
- On successful execution, the expected output is a prompt, which asks for a sentence and provides the probability of that sentence.

## Language Model Observations

### Language Model 1: 

- 4 gram model with Kneser-Ney smoothing on "Pride and Prejudice" corpus
- Avg perplexity for test data: 70.32
- Avg perplexity for train data: 5.88

### Language Model 2:

- 4 gram model with Witten-Bell smoothing on "Pride and Prejudice" corpus
- Avg perplexity for test data: 10.08
- Avg perplexity for train data: 2.20

### Language Model 3:

- 4 gram model with Kneser-Ney smoothing on "Ulysses" corpus
- Avg perplexity for test data: 139.88
- Avg perplexity for train data: 14.43

### Language Model 4:

- 4 gram model with Witten-Bell smoothing on "Ulysses" corpus
- Avg perplexity for test data: 9.52
- Avg perplexity for train data: 2.67

### Language Model 5:

- LSTM neural network model on "Pride and Prejudice" corpus
- Avg perplexity for test data: 45.68
- Avg perplexity for train data: 45.47

### Language Model 6:

- LSTM neural network model on "Ulysses" corpus
- Avg perplexity for test data: 42.15
- Avg perplexity for train data: 41.73
