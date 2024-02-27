import re
def tokenize(corpus):
    pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    list_sentences = re.split(pattern, corpus)
    url_pattern1 = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    url_pattern2 = r'www\.[^\s\.]+(?:\.[^\s\.]+)*(?:[\s\.]|$)'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    mention_pattern = "@\w+"
    hastag_pattern = "#[a-z0-9_]+"
    normal_pattern = "[a-zA-Z]+"
    number_pattern = "[0-9]+"
    tokens = []
    for sentence in list_sentences:
        sentence = sentence.lower()
        sentence = re.sub(url_pattern1, '<URL> ', sentence)
        sentence = re.sub(url_pattern2, '<URL> ', sentence)
        sentence = re.sub(email_pattern, '<MAILID> ', sentence)
        sentence = re.sub(hastag_pattern, '<HASHTAG> ', sentence)
        sentence = re.sub(mention_pattern, '<MENTION> ', sentence)
        sentence = re.sub(number_pattern, '<NUM> ', sentence)
        tokens.append(re.findall(r'\b\w+|[^\s\w<>]+|<\w+>', sentence))
    print(tokens)


if __name__ == '__main__':
    text = input("your text : ")
    # text = '''Is that what you mean? I am unsure.'''
    tokenize(text)
    print(tokenize)