import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import re
import random
import sys
import math
# nltk.download('punkt')

#maps 
ngram_context_counter = {}
ngram_continue_counter = {}
words_preceding_ngram = {}
words_prec_and_infront_ngram = {}
unique_words_preceding_ngram = {}
ngram_continue_counter_dict = {}

#helper funcitons 
def test_train_split(text ,n):
    text = re.sub(r"\n", ' ', text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    test_sentences = random.sample(sentences, n)
    train_sentences = [sentence for sentence in sentences if sentence not in test_sentences]
    return test_sentences, train_sentences

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

def generate_ngrams(tokens, n):
    tokens = (n-1)*['<START>']+tokens
    ngrams = [(tuple(tokens[i-p-1] for p in reversed(range(n-1))), tokens[i]) for i in range(n-1, len(tokens))]
    return ngrams

def gen_context_counter(ngrams):
    ngram_context = {}
    ngram_counter = {}

    for ngram in ngrams:
        if ngram in ngram_counter:
            ngram_counter[ngram] += 1
        else:
            ngram_counter[ngram] = 1
        
        prev_words, target_word = ngram
        
        if prev_words in ngram_context:
            ngram_context[prev_words].append(target_word)
        else:
            ngram_context[prev_words] = [target_word]
    
    return ngram_context, ngram_counter

def get_word_count(tokens):
    word_count = {}
    for word in tokens:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def gen_cont_counter(tokens, n):
    tokens = (n-1)*['<START>']+tokens
    ngrams_succ = [(tokens[i], tuple(tokens[i+p+1] for p in range(n-1))) for i in range(len(tokens) - n)]
    ngram_succ_counter = {}
    for ng in ngrams_succ:
        if ng in ngram_succ_counter:
            ngram_succ_counter[ng] += 1
        else:
            ngram_succ_counter[ng] = 1
    return ngram_succ_counter

def add_unk(word_count, tokens, thr):
    for i in range(len(tokens)):
        try:
            if(word_count[tokens[i]]<thr):
                tokens[i] = "<unk>"
        except KeyError:
            tokens[i] = "<unk>"
    return tokens

def key_ney(tokens, context, word, d, n, wc):
    word = word.lower()
    if n==1:
        
        unigram_prob = wc[word]/len(tokens)
        return unigram_prob
    try:
        if len(context) + 1 == n:
            if n>1:
                con = tuple(c.lower() for c in context[-(n-1):])
            else:
                con = ()
            
            if n in ngram_context_counter:
                ngram_context, ngram_counter = ngram_context_counter[n]
            else:
                ngrams = generate_ngrams(tokens, n)
                ngram_context, ngram_counter = gen_context_counter(ngrams)
                ngram_context_counter[n] = [ngram_context, ngram_counter]
        
            count_of_token = ngram_counter[(con, word)]
            
            count_of_context = float(len(ngram_context[con]))
            
            dinominator = count_of_context
            first_term = max(count_of_token-d,0)/count_of_context
            
        else:
            if n>1:
                context = tuple(c.lower() for c in context)
            else:
                context = ()
            
            if n+1 in ngram_continue_counter:
                succ_counter = ngram_continue_counter[n+1]
            else:
                succ_counter = gen_cont_counter(tokens, n+1)
                ngram_continue_counter[n+1] = succ_counter
            num_count = 0
            din_count = 0
            words = context[-(n-1):] + tuple([word])

            if n+1 in ngram_continue_counter_dict:
                words_pr_ngram, words_pr_inf_ngram, unique_words_pr_ngram = ngram_continue_counter_dict[n+1]
            else:
                succ_counter = gen_cont_counter(tokens, n+1)
                words_pr_ngram = {}
                words_pr_inf_ngram = {}
                unique_words_pr_ngram = {}
                for succ in succ_counter:
                    if succ[1] in words_pr_ngram:
                        words_pr_ngram[succ[1]] += succ_counter[succ]
                    else:
                        words_pr_ngram[succ[1]] = succ_counter[succ]
                    
                    if succ[1][:n-1] in words_pr_inf_ngram:
                        words_pr_inf_ngram[succ[1][:n-1]] += succ_counter[succ]
                        unique_words_pr_ngram[succ[1][:n-1]] += 1
                    else:
                        words_pr_inf_ngram[succ[1][:n-1]] = succ_counter[succ]
                        unique_words_pr_ngram[succ[1][:n-1]] = 1

                ngram_continue_counter_dict[n+1] = [words_pr_ngram, words_pr_inf_ngram, unique_words_pr_ngram] 

            num_count = words_pr_ngram[words]
            din_count = unique_words_pr_ngram[words[:n-1]]

            dinominator = din_count

            if dinominator == 0:
                return key_ney(tokens, context, word, 0, n-1, wc)
            first_term =  num_count/din_count
   
        #lambda calculation 
        if first_term != 0:

            if d==0:
                result = first_term
            else:
                lambda_val = d/dinominator
                result = first_term + lambda_val*key_ney(tokens, context, word, d,  n-1, wc)

        else:
       
            d = 0.75
            lambda_val = d/dinominator
            result = lambda_val*key_ney(tokens, context, word, d,  n-1, wc)
        return result
  
    except KeyError:
        result = key_ney(tokens, context, word, 0.75, n-1, wc)
        return result

def wittenbell(tokens, ngram, n, wc):
    context, word = ngram
    if n+1 in ngram_continue_counter:
        succ_counter = ngram_continue_counter[n+1]
    else:
        succ_counter = gen_cont_counter(tokens, n+1)
        ngram_continue_counter[n+1] = succ_counter

    if n+1 in ngram_continue_counter_dict:
        words_pr_ngram, words_pr_inf_ngram, unique_words_pr_ngram = ngram_continue_counter_dict[n+1]
    else:
        succ_counter = gen_cont_counter(tokens, n+1)
        words_pr_ngram = {}
        words_pr_inf_ngram = {}
        unique_words_pr_ngram = {}
        for succ in succ_counter:
            if succ[1] in words_pr_ngram:
                words_pr_ngram[succ[1]] += succ_counter[succ]
            else:
                words_pr_ngram[succ[1]] = succ_counter[succ]
            
            if succ[1][:n-1] in words_pr_inf_ngram:
                words_pr_inf_ngram[succ[1][:n-1]] += succ_counter[succ]
                unique_words_pr_ngram[succ[1][:n-1]] += 1
            else:
                words_pr_inf_ngram[succ[1][:n-1]] = succ_counter[succ]
                unique_words_pr_ngram[succ[1][:n-1]] = 1

        ngram_continue_counter_dict[n+1] = [words_pr_ngram, words_pr_inf_ngram, unique_words_pr_ngram] 

    num_count = 0
    din_count = 0

    words = context[-(n-1):] + tuple([word])
    c_kn_num, c_kn_din = 0, 0
    no_words_following = 0
    ng_word = ngram[0] + (ngram[1],)


    c_kn_num = words_pr_ngram[ng_word] if ng_word in words_pr_ngram else 0
    c_kn_din = words_pr_inf_ngram[ng_word[:n-1]] if ng_word[:n-1] in words_pr_inf_ngram else 0
    no_words_following = unique_words_pr_ngram[ng_word[:n-1]] if ng_word[:n-1] in unique_words_pr_ngram else 0
        

    if (context == ()): return (c_kn_num + no_words_following)/(c_kn_din + no_words_following)

    lower_gram =  (context[1:], word)

    num = c_kn_num + no_words_following*wittenbell(tokens, lower_gram, n-1, wc)
    din = c_kn_din + no_words_following
    if din ==0: return wittenbell(tokens, lower_gram, n-1, wc)
    return num/din

def get_kneser_ney_perplexity(test_tokens, tokens, wc):
    perplexity = 1
    n=0
    log_sum = 0
    if len(test_tokens) == 0: return 0
    if(len(test_tokens)<4):
        context = tuple(test_tokens[:-1])
        word = test_tokens[-1]
        prob = key_ney(tokens, context, word, 0.75, len(test_tokens), wc)
        n += 1
        log_sum +=math.log(prob)
    else:
        for i in range(len(test_tokens)-3):
            con = tuple(test_tokens[i:i+3])
            # print(con)
            word = test_tokens[i+3]
            prob = key_ney(tokens, con, word, 0.75, 4, wc)
            n+=1
            log_sum +=math.log(prob)
    perplexity = math.exp(-1/n*log_sum)
    # print(perplexity, n)
    return perplexity

def get_witten_bell_perplexity(test_tokens, tokens, wc):
    perplexity = 1
    n=0
    log_sum = 0
    if len(test_tokens) == 0: return 0
    if(len(test_tokens)<4):
        context = tuple(test_tokens[:-1])
        word = test_tokens[-1]
        ng = (context, word)
        prob = wittenbell(tokens, ng, len(test_tokens), wc)
        n += 1
        log_sum +=math.log(prob)
    else:
        for i in range(len(test_tokens)-3):
            context = tuple(test_tokens[i:i+3])
            # print(con)
            word = test_tokens[i+3]
            ng = (context, word)
            prob = wittenbell(tokens, ng, 4, wc)
            n+=1
            log_sum +=math.log(prob)
    perplexity = math.exp(-1/n*log_sum)
    return perplexity


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Correct usage format is python3 language_model.py <smoothing> <corpus>")
        exit()
    smoothing_technique = str(sys.argv[1])
    corpus = sys.argv[2]

    if smoothing_technique != 'k' and smoothing_technique != 'w':
        print("Enter smoothing technique k (Kneser-Ney) or w (Witten-Bell)")
        exit()
    
    #add checking for file 


    #reading the text file 
    with open(corpus, "r") as file:
        text = file.read()

    #splitting test and train set 
    test_sentences, train_sentences = test_train_split(text, 1000)

    #tokenizing the train sentences
    tokens = tokenize("".join(train_sentences))

    #getting word count and adding <unk>
    wc = get_word_count(tokens)
    updated_tokens = add_unk(wc, tokens, 2)
    new_wc = get_word_count(updated_tokens)

    # #kneser-ney calcuation 
    # with open("Ulysses_LM3_k_test-perplexity.txt", "w") as f:
    #     k_sum = 0
    #     for sentence in test_sentences:
    #         test_tokens = tokenize("".join(sentence))
    #         test_tokens = add_unk(new_wc, test_tokens, 2)
    #         perplexity = get_kneser_ney_perplexity(test_tokens, updated_tokens, new_wc)
    #         k_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_k_test_perplexity = k_sum/len(test_sentences)
    #     line = "Avg perplexity: " + str(avg_k_test_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()

    # with open("Ulysses_LM3_k_train-perplexity.txt", "w") as f:
    #     k_sum = 0
    #     for sentence in train_sentences:
    #         test_tokens = tokenize("".join(sentence))
    #         test_tokens = add_unk(new_wc, test_tokens, 2)
    #         perplexity = get_kneser_ney_perplexity(test_tokens, updated_tokens, new_wc)
    #         k_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_k_train_perplexity = k_sum/len(train_sentences)
    #     line = "Avg perplexity: " + str(avg_k_train_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()


    # #witten-bell calculation
    # with open("Ulysses_LM4_w_test-perplexity.txt", "w") as f:
    #     w_sum = 0
    #     for sentence in test_sentences:
    #         test_tokens = tokenize("".join(sentence))
    #         test_tokens = add_unk(new_wc, test_tokens, 2)
    #         perplexity = get_witten_bell_perplexity(test_tokens, updated_tokens, new_wc)
    #         w_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_w_test_perplexity = w_sum/len(test_sentences)
    #     line = "Avg perplexity: " + str(avg_w_test_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()

    # with open("Ulysses_LM4_w_train-perplexity.txt", "w") as f:
    #     w_sum = 0
    #     for sentence in train_sentences:
    #         test_tokens = tokenize("".join(sentence))
    #         test_tokens = add_unk(new_wc, test_tokens, 2)
    #         perplexity = get_witten_bell_perplexity(test_tokens, updated_tokens, new_wc)
    #         w_sum += perplexity
    #         line = sentence + ": " + str(perplexity) + "\n"
    #         f.writelines(line)
    #     avg_w_train_perplexity = w_sum/len(train_sentences)
    #     line = "Avg perplexity: " + str(avg_w_train_perplexity) + "\n\n"
    #     f.seek(0)
    #     f.writelines(line)
    # f.close()

    # print("Kneser-Ney Peplexity for test data: ",avg_k_test_perplexity)
    # print("Kneser-Ney Peplexity for train data: ",avg_k_train_perplexity)
    # print("Witten-bell Peplexity for test data: ",avg_w_test_perplexity)
    # print("Witten-bell Peplexity for train data: ",avg_w_train_perplexity)

    quit = False
    while not quit:
        in_sentence = input("Enter the sentence: ")
        if smoothing_technique == "k":
            test_tokens = tokenize("".join(in_sentence))
            test_tokens = add_unk(new_wc, test_tokens, 2)
            perplexity = get_kneser_ney_perplexity(test_tokens, updated_tokens, new_wc)
            print("Perplexity = ",perplexity)
        else:
            test_tokens = tokenize("".join(in_sentence))
            test_tokens = add_unk(new_wc, test_tokens, 2)
            perplexity = get_witten_bell_perplexity(test_tokens, updated_tokens, new_wc)
            print("Perplexity = ",perplexity)
        c = input("Continue(y/n)")
        if c != 'y':
            quit = True





    

