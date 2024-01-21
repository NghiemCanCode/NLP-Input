import numpy as np
from collections import OrderedDict
import math
class MyTF_IDF:
    def __init__(self):
        self.IDF = None

    def fit(self, corpus):
        corpus_len = len(corpus)
        word_dict = {}

        if type(corpus) != list:
            raise TypeError("The Corpus must be a list")
        for text in corpus:
            tmp_text_arr = text.split()
            mini_dict = {}
            for word in tmp_text_arr:
                if word not in mini_dict:
                    mini_dict[word] = 1
            for word in mini_dict:
                if word in word_dict:
                    word_dict[word] += mini_dict[word]
                else:
                    word_dict[word] = mini_dict[word]
        for word in word_dict:
            word_dict[word] = math.log10(corpus_len / word_dict[word])
        self.IDF = dict(OrderedDict(sorted(word_dict.items())))
    def transform(self, text_list):
        # warning array here

        if self.IDF == None or self.IDF == []:
            raise VocabularyError("You must fit vector with copurs")
        vocab_arr = list(self.IDF.keys())
        vector_list = np.zeros([len(text_list), len(vocab_arr)])

        for index, text in enumerate(text_list):
            word_arr = text.split()

            for word in word_arr:
                if word not in vocab_arr:
                    raise CorpusError("The word not in vocab")
                vector_list[index, vocab_arr.index(word)] += 1

            vector_list[index:] /= np.max(vector_list[index, :])
            vector_list[index:] *= np.array(list(self.IDF.values()))

        return vector_list

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)