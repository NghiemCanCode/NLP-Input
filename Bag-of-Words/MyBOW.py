import numpy as np

class MyBOW:
    def __init__(self):
        self.vocab = None

    def fit(self, corpus):
        text_flatten = ""
        for text in corpus:
            text_flatten = text_flatten + " " + text
        self._word_count(text_flatten)
    def _word_count(self, corpus):
        word_dict = {}
        word_arr = corpus.split()
        for word in word_arr:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

        list_vocab = list(word_dict.keys())

        self.vocab = sorted(list_vocab)

    def transfer(self, text_list):

        # warning array here

        if self.vocab == None or self.vocab == []:
            raise VocabularyError("You must fit vector with copurs")

        vector_list = np.zeros([len(text_list), len(self.vocab)])

        for index, text in enumerate(text_list):
           word_arr = text.split()

           for word in word_arr:
               if word not in self.vocab:
                   raise CorpusError("The word not in vocab")
               vector_list[index, self.vocab.index(word)] += 1

        return vector_list

    def fit_transfer(self, corpus):
        self.fit(corpus)
        return self.transfer(corpus)