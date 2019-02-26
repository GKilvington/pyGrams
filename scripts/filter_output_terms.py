import zipfile
from gensim.models import KeyedVectors
from scripts.tfidf_wrapper import TFIDF

class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, file_path, file_name, model=None, binary_vect = list(),
                 distance_vect = list(), binary=False, threshold = None):
        print("filter terms")
        self.__user_ngrams = user_ngrams
        print('user ngrams = ', self.__user_ngrams)
        self.__tfidf_ngrams = tfidf_ngrams
        self.__file_path = file_path
        self.__file_name = file_name
        self.binary_vect = binary_vect
        self.distance_vect = distance_vect
        self.model = model
        self.binary = binary
        self.threshold = threshold

        print('tfidf ngrams = ', self.__tfidf_ngrams)
        self.load_fasttext_model()

        if not binary:
            for term in self.__tfidf_ngrams:
                print('term = ', term)
                compare = []
                for ind_term in term.split():
                    print('ind_term = ', ind_term)
                    for user_term in self.__user_ngrams:
                        print('user_term =', user_term)
                        try:
                            j = self.calculate_distances(ind_term, user_term)
                            print('j=', j)
                            compare.append(j)
                        except:
                            continue
                print('compare =', compare)
                if compare:
                    x = min(float(s) for s in compare)
                    print(x)
                    if x > self.threshold:
                        self.distance_vect.append(x)
                    else:
                        self.distance_vect.append(0)
                else:
                    self.distance_vect.append(0)
        print(self.distance_vect)

    def load_fasttext_model(self):
        fasttext = zipfile.ZipFile(self.__file_path, 'r')
        file = fasttext.open(self.__file_name)
        self.model = KeyedVectors.load_word2vec_format(file)

    def calculate_distances(self, i, p):
        return self.model.similarity(i, p)
        # populate this one: self.__ngrams_weights_vect


    #embeddings best to be blended in the mask! ie. data[i] *= cosine_distance between domain words and ngram_average

    # @property
    # def ngrams_weights_vect(self):
    #     return self.__ngrams_weights_vect
    #
    # def ngrams_binary_vect(self):
    #     return self.__ngrams_binary_vect


# if self.filter_output:
#     filter_outputs = input("Please specify interested words: ")
#     filter_outputs_list = filter_outputs.split(",")
#     from gensim.models import KeyedVectors
#     import zipfile
#     fasttext = zipfile.ZipFile('models/wiki-news-300d-1M.vec.zip', 'r')
#     file = fasttext.open('wiki-news-300d-1M.vec')
#     model = KeyedVectors.load_word2vec_format(file)
#     ngrams_scores_slice_1 = []
#     for i in ngrams_scores_slice:
#         for g in i[1].split():
#             for p in filter_outputs_list:
#                 try:
#                     j = model.similarity(g, p)
#                     if j > 0.40:
#                         ngrams_scores_slice_1.append(i)
#                 except:
#                     continue
#     ngrams_scores_slice = ngrams_scores_slice_1