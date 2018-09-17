import string
import nltk
import re
# import numpy as np
# from nltk.corpus import words as nltk_words
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Tokenizer:
    def __init__(self, text = None, delimiter = ' ', stop_words = []):
        # Initialize helper variables
        self.delimiter = delimiter
        self.stop_words = stop_words
        self.__punc_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        self.__lemmatizer = nltk.stem.WordNetLemmatizer()

        if not text is None:
            self.text = text
            self.sentences = nltk.tokenize.sent_tokenize(text)
            self.sentence_tokens = []

    def _delimitize(self, sentence, delimiter = ' '):
        return re.sub(r'([a-zA-Z])\-([a-zA-Z])', r'\1' + self.delimiter + r'\2', sentence)

    def _unlatex(self, sentence):
        return re.sub(r'([^\\])\$(.*?)([^\\])\$', "", sentence)

    def _lemmatize(self, word_tokens):
        lemma_tokens = []
        for word in word_tokens:
            verb_lemma = self.__lemmatizer.lemmatize(word, 'v')
            if verb_lemma == word:
                noun_lemma = self.__lemmatizer.lemmatize(word, 'n')
                lemma_tokens.append(noun_lemma)
            else:
                lemma_tokens.append(verb_lemma)
        return lemma_tokens

    def _tokenize_sentence(self, sentence):
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.lower()
        sentence = self._delimitize(sentence)
        sentence = self._unlatex(sentence)
        # sentence = re.sub('\\\\[\'"]', '', sentence)
        sentence = sentence.translate(self.__punc_trans)
        raw_word_tokens = nltk.tokenize.word_tokenize(sentence)
        word_tokens = []
        for word in raw_word_tokens:
            if not (len(word) == 0 or word.isnumeric() or word in self.stop_words):
                word_tokens.append(word)
        return word_tokens

    def load(self, text):
        self.text = text
        self.sentences = nltk.tokenize.sent_tokenize(text)

    def tokenize(self, lemmatize = False):
        "Remove LaTeX tags, and extra whitespace, stopwords, and non-alphanumeric \
        characters. Make all lower case."
        assert isinstance(lemmatize, bool)
        try:
            sentences = nltk.tokenize.sent_tokenize(self.text)
        except TypeError:
            print("Text must be loaded into Tokenizer.")
        self.sentence_tokens = []
        for sentence in sentences:
            word_tokens = self._tokenize_sentence(sentence)
            if lemmatize:
                word_tokens = self._lemmatize(word_tokens)
            self.sentence_tokens.append(word_tokens)


# class ReducedCountVectorizer(CountVectorizer):
#     def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
#                        lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None,
#                        token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1,
#                        max_features=None, vocabulary=None, binary=False, dtype=np.int64, norm='l2',
#                        use_idf=True, smooth_idf=True, sublinear_tf=False):
#         super().__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer,
#                          stop_words, token_pattern, ngram_range, analyzer, max_df, min_df, max_features,
#                          vocabulary, binary, dtype)
#         self.__tfidf_vectorizer = TfidfVectorizer(input, encoding, decode_error, strip_accents, lowercase,
#                                                   preprocessor, tokenizer, analyzer, stop_words, token_pattern,
#                                                   ngram_range, max_df, min_df, max_features, vocabulary, binary,
#                                                   dtype, norm, use_idf, smooth_idf, sublinear_tf)

#     def fit(self, *args, **kwargs):
#         self.__tfidf_vectorizer.fit(*args, **kwargs)

#         vocabulary = list(self.__tfidf_vectorizer.vocabulary_.keys())
#         for term in tuple(vocabulary):
#             if term in nltk_words.words('en'):
#                 vocabulary.remove(term)

#         self.set_params(vocabulary = iter(vocabulary))
#         super().fit(*args, **kwargs)