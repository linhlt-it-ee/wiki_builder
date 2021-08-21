from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
import numpy as np
def split_tokenizer(text):
    return text.split()
def get_pmi(prj_path, dataset,doc_content_list, vocab):
    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in doc_content_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # vocab = list(word_set)
    vocab_size = len(vocab)

    word_doc_list = {}

    for i in range(len(doc_content_list)):
        doc_words = doc_content_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open(prj_path + 'data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in doc_content_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    word_pair_count_wth_id = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                    # word_pair_count_wth_id[word_i+','+word_j]+=1
                else:
                    word_pair_count[word_pair_str] = 1
                    # word_pair_count_wth_id[word_i + ',' + word_j] =1
    # row = []
    # col = []
    # weight = []

    # pmi as weights

    num_window = len(windows)
    pmi_word_word=np.zeros((len(vocab),len(vocab)))
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        pmi_word_word[i][j]=pmi
        # row.append(train_size + i)
        # col.append(train_size + j)
        # weight.append(pmi)
    # print("word pair count",word_pair_count_wth_id)

    print("word id map", len(word_id_map.keys()))
    return pmi_word_word

def get_tfidf_score(texts):
    vectorizer = TfidfVectorizer(tokenizer=split_tokenizer)
    X = vectorizer.fit_transform(texts)
    print(vectorizer.get_feature_names())
    tf_vocab=vectorizer.vocabulary_
    sorted_tfvocab={k: v for k, v in sorted(tf_vocab.items(), key=lambda item: item[1])}
    return X, sorted_tfvocab
doc_content_list = []
prj_path="/home/s4616573/code/tf_ner/text_gcn/"
dataset="i2b2"
f = open(prj_path+'data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
tf_idf_matrix, sorted_tfvocab=get_tfidf_score(doc_content_list)
pmi_matrix=get_pmi(prj_path, dataset,doc_content_list,len(doc_content_list),list(sorted_tfvocab.keys()))
# file_util.dump(X,simulation_folder+"/i2b2_train_word_sen_matrix.pck")