import os
import sys
import re
# import requests
# import mwparserfromhell
import nltk
import numpy as np
from bs4 import BeautifulSoup
# import scipy.spatial.distance as sdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.externals import joblib
from gensim.models import KeyedVectors
from flask import Flask, request, render_template
from flask_cors import CORS

# Local
## Allow local relative imports
module_path = os.path.abspath('..')
include_path = os.path.join(module_path, 'include')
models_path = os.path.join(module_path, 'models')
if include_path not in sys.path:
    sys.path.append(include_path)

from my_nlp import Tokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def load_embedding(path):
    return KeyedVectors.load(path)


def shift_sup_left(text):
    temp_text = str(text)
    shifted_text = re.sub(r'\.(<sup.*?/sup>)', r'\1.', temp_text)
    while shifted_text != temp_text:
        temp_text = shifted_text
        shifted_text = re.sub(r'\.(<sup.*?/sup>)', r'\1.', temp_text)
    return shifted_text


def shift_sup_right(text):
    temp_text = str(text)
    shifted_text = re.sub(r'(<sup.*?/sup>)\.', r'.\1', temp_text)
    while shifted_text != temp_text:
        temp_text = shifted_text
        shifted_text = re.sub(r'(<sup.*?/sup>)\.', r'.\1', temp_text)
    return shifted_text


def mean_filtered(embed, doc, vocab_weights = None):
    embed_filtered = [x for x in doc.split() if x in embed.vocab]
    if len(embed_filtered) > 0:
        if vocab_weights is None:
            return embed[embed_filtered].mean(axis = 0)
        else:
            weight_filtered = [x for x in embed_filtered if x in vocab_weights]
            if len(weight_filtered) > 0:
                weights = np.array([vocab_weights[x] for x in weight_filtered])
                return weights.dot(embed[weight_filtered]) / weights.sum()
            else:
                return np.zeros(embed.vector_size, dtype = np.float32)
    else:
        return np.zeros(embed.vector_size, dtype = np.float32)


def logreg_distance(logreg, A, B):
    r = np.empty((A.shape[0], B.shape[0], A.shape[1] + B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            r[i, j] = np.concatenate([A[i], B[j]], axis = 0)
    r = r.reshape(-1, A.shape[1] + B.shape[1])
    r = logreg.predict_proba(r)[:, 1].reshape(A.shape[0], B.shape[0])
    return r


# sess = requests.Session()
# wapi_url = "https://en.wikipedia.org/w/api.php"

tok = Tokenizer()


app = Flask(__name__)
CORS(app)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/api/<uuid>', methods = ["GET", "POST"])
def apply_model(uuid):
    content = request.get_json(force = True)

    # origin_title = content['origin']
    # target_title = content['target']

    # if (origin_title is None) or (target_title is None):
    #     return ""

    # wapi_params = {
    #     'action': "query",
    #     'maxlag': 5,
    #     'prop': "revisions",
    #     'titles': '|'.join([origin_title, target_title]),
    #     'rvprop': "content",
    #     'rvslots': "main",
    #     'format': "json"
    # }

    # data = sess.get(url = wapi_url, params = wapi_params).json()
    # print(origin_title, target_title)

    # norm_origin_title = origin_title
    # norm_target_title = target_title
    # for title in data['query']['normalized']:
    #     if title['from'] == origin_title:
    #         norm_origin_title = title['to']
    #     elif title['from'] == target_title:
    #         norm_target_title = title['to']

    # for page in data['query']['pages'].values():
    #     if page['title'] == norm_origin_title:
    #         origin_extract = page['revisions'][0]['slots']['main']['*']
    #         parsed_origin_extract = mwparserfromhell.parse(origin_extract.replace('\n', ' ')).strip_code()
    #         origin_sentences = nltk.tokenize.sent_tokenize(parsed_origin_extract)
    #         origin_sentence_tokens = tok.load(parsed_origin_extract).tokenize(lemmatize = True).sentence_tokens
    #     elif page['title'] == norm_target_title:
    #         target_extract = page['revisions'][0]['slots']['main']['*']
    #         parsed_target_extract = mwparserfromhell.parse(target_extract.replace('\n', ' ')).strip_code()
    #         target_sentences = nltk.tokenize.sent_tokenize(parsed_target_extract)
    #         target_sentence_tokens = tok.load(parsed_target_extract).tokenize(lemmatize = True).sentence_tokens

    # title_tokens = tok.load(norm_origin_title).tokenize(lemmatize = True).sentence_tokens[0]
    # hover_text = '\n'.join([target_sentences[i] for i in range(len(target_sentence_tokens)) if all(x in target_sentence_tokens[i] for x in title_tokens)])
    
    # return str.encode(hover_text)

    origin_title = content['origin_title']
    target_title = content['target_title']

    origin_content = shift_sup_left(content['origin_content'])
    # origin_content = BeautifulSoup(content['origin_content'], 'html5lib')
    # for x in origin_content.find_all("sup", {'class': 'reference'}):
    #     x.decompose()
    # origin_content = origin_content.body.decode_contents()

    target_content = shift_sup_left(content['target_content'])
    # target_content = BeautifulSoup(content['target_content'], 'html5lib')
    # for x in target_content.find_all("sup", {'class': 'reference'}):
    #     x.decompose()
    # target_content = target_content.body.decode_contents()

    origin_context_a = shift_sup_left(content['origin_context_a'])
    # origin_context_a = BeautifulSoup(content['origin_context_a'], 'html5lib')
    # for x in origin_context_a.find_all("sup", {'class': 'reference'}):
    #     x.decompose()
    # origin_context_a = origin_context_a.body.decode_contents()

    origin_context_p = shift_sup_left(content['origin_context_p'])
    # origin_context_p = BeautifulSoup(content['origin_context_p'], 'html5lib')
    # for x in origin_context_p.find_all("sup", {'class': 'reference'}):
    #     x.decompose()
    # origin_context_p = origin_context_p.body.decode_contents()

    origin_context = origin_context_p
    for origin_context_p_sentence in nltk.sent_tokenize(origin_context_p):
        if origin_context_a in origin_context_p_sentence:
            origin_context = origin_context_p_sentence.strip()

    origin_title_tokens = tok.load(origin_title).word_tokenize(lemmatize = True).word_tokens

    origin_sentence_htmls = []
    origin_sentence_tokens = []
    # origin_context_sentence_ind = 0
    # i = 0
    for p_split in origin_content.split('\n'):
        for origin_sentence_html in nltk.sent_tokenize(p_split):
            origin_sentence_html = origin_sentence_html.strip()
            if origin_sentence_html == origin_context:
                # origin_context_sentence_ind = i
                origin_sentence_soup = BeautifulSoup(origin_sentence_html, 'html5lib').body
                origin_sentence_text = re.sub('\[.*?\]', '', origin_sentence_soup.get_text().replace(u'\xa0', u' '))
                origin_word_tokens = tok.load(origin_sentence_text).word_tokenize(lemmatize = True).word_tokens
                if len(origin_word_tokens) > 0:
                    origin_sentence_htmls.append(shift_sup_right(origin_sentence_soup.decode_contents()))
                    origin_sentence_tokens.append(' '.join(origin_word_tokens))
                    break
        # i += 1

    target_sentence_htmls = []
    target_sentence_tokens = []
    for p_split in target_content.split('\n'):
        for target_sentence_html in nltk.sent_tokenize(p_split):
            target_sentence_soup = BeautifulSoup(target_sentence_html, 'html5lib').body
            target_sentence_text = re.sub('\[.*?\]', '', target_sentence_soup.get_text().replace(u'\xa0', u' '))
            target_word_tokens = tok.load(target_sentence_text).word_tokenize(lemmatize = True).word_tokens
            if len(target_word_tokens) > 0:
                target_sentence_htmls.append(shift_sup_right(target_sentence_soup.decode_contents()))
                target_sentence_tokens.append(' '.join(target_word_tokens))

    # hover_text = '\n'.join([target_sentence_htmls[i] for i in range(len(target_sentence_tokens)) if ' '.join(origin_title_tokens) in ' '.join(target_sentence_tokens[i])])

    # max_df = 0.9
    # min_df = 1

    # max_features = 1000

    # min_n_gram = 1
    # max_n_gram = 1

    # # print(origin_sentence_tokens)

    # tfidf_vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df,
    #                                    max_features = max_features,
    #                                    ngram_range = (min_n_gram, max_n_gram),
    #                                    stop_words = 'english').fit([' '.join(x) for x in origin_sentence_tokens])

    # origin_sentence_vectors = tfidf_vectorizer.transform([' '.join(x) for x in origin_sentence_tokens]).toarray()
    # target_sentence_vectors = tfidf_vectorizer.transform([' '.join(x) for x in target_sentence_tokens]).toarray()

    # top_match_indices = sdist.cdist(origin_sentence_vectors, target_sentence_vectors, metric = 'cosine')[origin_context_sentence_ind].argsort()#[::-1]

    if len(origin_sentence_tokens) == 0:
        hover_text = ' '.join(target_sentence_htmls[:10])
    else:
        origin_sentence_vectors = wiki2vec_vectorizer(origin_sentence_tokens)#np.array([mean_filtered(embed, sent) for sent in origin_sentence_tokens])
        target_sentence_vectors = wiki2vec_vectorizer(target_sentence_tokens)#np.array([mean_filtered(embed, sent) for sent in target_sentence_tokens])

        top_match_indices = (1 - cosine_similarity(origin_sentence_vectors, target_sentence_vectors)[0]).argsort()

        n_top_matches = 10
        # sent_range = 2

        # hover_text = '\n'.join([' '.join(target_sentence_htmls[i - sent_range:i + sent_range + 1]) for i in top_match_indices[:n_top_matches]])
        # hover_text = '<br>---<br>'.join([target_sentence_htmls[i] for i in top_match_indices[:n_top_matches]])
        hover_text = '<br>---<br>'.join([target_sentence_htmls[i] for i in top_match_indices[:n_top_matches]])
    
    return str.encode(hover_text)


print("Loading model...")
wiki2vec_embed = KeyedVectors.load(models_path + '/wiki2vec/en.model.kv')
wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x), signature = '()->(n)')
# wiki2vec_tfidf_embed = joblib.load(models_path + '/tfidf/enwiki-latest-all-wiki2vec_tfidf_embed.joblib')
# wiki2vec_idf = dict(zip(wiki2vec_tfidf_embed.get_feature_names(), wiki2vec_tfidf_embed.idf_))
# tfidf_wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x, vocab_weights = wiki2vec_idf), signature = '()->(n)')
print("Model successfully loaded.")

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 5000)
