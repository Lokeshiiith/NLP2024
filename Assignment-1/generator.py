import re
import random
import sys
import math



def test_train_split(corpus, n):
    # remove new line
    corpus = corpus.replace('\n', ' ')
    # split into sentences
    sentences = re.split(r'(?<=[.!?]) +', corpus)
    test_sentences = random.sample(sentences, n)
    train_sentences = [sentence for sentence in sentences if sentence not in test_sentences]
    return test_sentences, train_sentences


def tokenize(text):
    url_pattern1 = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    url_pattern2 = r'www\.[^\s\.]+(?:\.[^\s\.]+)*(?:[\s\.]|$)'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    mention_pattern = "@\w+"
    hastag_pattern = "#[a-z0-9_]+"
    normal_pattern = "[a-zA-Z]+"
    number_pattern = "[0-9]+"
    tokens = []
    text = text.lower()
    text = re.sub(url_pattern1, '<URL> ', text)
    text = re.sub(url_pattern2, '<URL> ', text)
    text = re.sub(email_pattern, '<MAILID> ', text)
    text = re.sub(hastag_pattern, '<HASHTAG> ', text)
    text = re.sub(mention_pattern, '<MENTION> ', text)
    text = re.sub(number_pattern, '<NUM> ', text)
    tokens = re.findall(r'\b\w+|[^\s\w<>]+|<\w+>', text)
    return tokens

def perform_cleaning(text):
    # remove comma, extra spaces, and punctuations
    text = re.sub(r'[,!?;-]+', '', text)
    if text.endswith('.'):
            text = text[:-1]#removing last dot also
    return text
def linear_interpolation(trigram, unigram_prob, bigram_prob, trigram_prob):
    lambda1 = 0.4
    lambda2 = 0.3
    lambda3 = 0.3
    unigram_probability = unigram_prob.get(trigram[-1], 0)
    bigram_probability = bigram_prob.get(tuple(trigram[-2:]), 0)
    trigram_probability = trigram_prob.get(tuple(trigram[-3:]), 0)

    unigram_probability = lambda3 * unigram_probability  
    if unigram_probability == 0:
        unigram_probability = 0.00001
    
    trigram_probability = lambda1 * trigram_probability
    if trigram_probability == 0:
        trigram_probability = 1/len(trigram_prob)

    bigram_probability = lambda2 * bigram_probability
    if bigram_probability == 0:
        bigram_probability = 1/len(bigram_prob)

    interpolated_prob = trigram_probability + bigram_probability + unigram_probability
    return interpolated_prob
def perplexity(sentence, unigram_prob, bigram_prob, trigram_prob):
    tokens = tokenize(sentence)
        # in this tuple add <start> <start> at the start of the sentence and <end> at the end of the sentence
    tokens = ('<START>', '<START>',) + tuple(tokens) + ('<END>',)
    # break  
    log_probability_sum = 0.0
    trigram_count = 1
    for i in range(len(tokens)-3):
        trigram = tuple(tokens[i:i+3])
        trigram_count += 1
        temp_prob = math.log(linear_interpolation(trigram, unigram_prob, bigram_prob, trigram_prob))
        # log_probability_sum += math.log(linear_interpolation(trigram, unigram_prob, bigram_prob, trigram_prob))
        log_probability_sum += temp_prob

    sentence_perplexity = math.exp(-(log_probability_sum / trigram_count))
    return sentence_perplexity

def PerformNgram(corpus, n):
    pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    list_sentences = re.split(pattern, corpus)
    ngrams = {}
    for sentence in list_sentences:
        tokens = tokenize(sentence)
        # sentence = (n-1)*"<START> "+ sentence
        for i in range(len(tokens)-n+1):
            temp = tuple(tokens[j] for j in range(i, i+n))  # Convert list to tuple
            if temp in ngrams:
                ngrams[temp] += 1
            else:
                ngrams[temp] = 1
            
    return ngrams
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Correct usage format is python3 generator.py <smoothing> <corpus><numbers>")
        print("where <smoothing> g  is (good turing) and i is (interpolation)")
        print("and <corpus> is the path to the corpus file <corpus> p is (./corpus/Pride and Prejudice - Jane Austen.txt)")
        print("<corpus> u is ./corpus/Ulysses  James Joyce.txt")
        exit()
    smoothing_technique = str(sys.argv[1])
    k = int(sys.argv[3])
    if smoothing_technique != 'g' and smoothing_technique != 'i':
        print("Enter smoothing technique  g  for  (good turing) OR i for (interpolation)")
        exit()
    corpus = str(sys.argv[2])
    if corpus != 'p' and corpus != 'u' and corpus != './corpus/Pride and Prejudice - Jane Austen.txt' and corpus != './corpus/Ulysses  James Joyce.txt':
        print("Enter corpus path <corpus> p or (./corpus/Pride and Prejudice - Jane Austen.txt)")
        print("<corpus> u or ./corpus/Ulysses  James Joyce.txt")
        exit()
    if corpus == 'p':
        corpus = './corpus/Pride and Prejudice - Jane Austen.txt'
    elif corpus == 'u':
        corpus = './corpus/Ulysses  James Joyce.txt'
    
    # splitting the sentence
    with open(corpus, 'r', encoding='utf-8') as f:
        text = f.read()
    test_sentences, train_sentences = test_train_split(text, 1000)
    quit = False
    while not quit:
        in_sentence = str(input("Enter the sentence: "))
        if smoothing_technique == "i":
            test_tokens = tokenize("".join(in_sentence))
            unigram_count = PerformNgram(" ".join(train_sentences), 1)
            bigram_count = PerformNgram(" ".join(train_sentences), 2)
            trigram_count = PerformNgram(" ".join(train_sentences), 3)
            unigram_prob = {}
            bigram_prob = {}
            trigram_prob = {}
            for key, value in unigram_count.items():
                unigram_prob[key] = value / len(unigram_count)

            for bigram, count in bigram_count.items():
                w1 = bigram[0]
                w1_token = (w1,)
                bigram_prob[bigram] = count / unigram_count[w1_token]

            for trigram, w1_w2_w3 in trigram_count.items():
                w1_w2_token = (trigram[0], trigram[1],)
                trigram_prob[trigram] = w1_w2_w3 / bigram_count[w1_w2_token]

            token = tokenize(in_sentence)
            w1 = token[-2:]
            w1 = tuple(w1,)
            word_probabilities = {}
            for sentence in train_sentences:
                # Split the sentence into words
                words = tokenize(sentence)
                
                # Extract the last two words and convert them into a tuple
                for eachword in words:
                    w = (w1,eachword)
                    w = w[0] + (w[1],)
                    prob = linear_interpolation(w,unigram_prob, bigram_prob, trigram_prob)
                    word_probabilities[eachword] = prob
            sorted_word_probabilities = sorted(word_probabilities.items(), key=lambda x: x[1], reverse=True)
            for word, prob in sorted_word_probabilities[:k]:
                print(f"Probability: {prob}, Word: {word}")
        else:
            print("Not completed:")
        c = input("Continue(y/n) : ")
        if c != 'y':
            quit = True

    