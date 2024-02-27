import re
import collections

def perform_cleaning(text):
    # remove comma, extra spaces, and punctuations
    text = re.sub(r'[,!?;-]+', '', text)
    if text.endswith('.'):
            text = text[:-1]
    return text
def PerformNgram(corpus, n):
    pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    list_sentences = re.split(pattern, corpus)
    ngrams = collections.defaultdict(int)
    for sentence in list_sentences:
        sentence = perform_cleaning(sentence)
        tokens = re.split("\\s+", sentence)
        for i in range(len(tokens)-n+1):
            temp = [tokens[j] for j in range(i, i+n)]
            ngram = (" ".join(temp))
            # ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
    return ngrams

if __name__ == '__main__':
    ngram = []
    corpus = '''Hi, I am Lokesh, email id is lokeshsharma123456@gmail.com. My rollnumber is 2022201041. Mywebsite is www.abc.co.in. I
            am using #python. My twitter id is @lokeshsharma.'''
    ngrams = PerformNgram(corpus, 2)
    print(ngrams)