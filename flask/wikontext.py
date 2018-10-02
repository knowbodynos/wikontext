import os
import sys
import re
# import requests
# import mwparserfromhell
import nltk
import numpy as np
from bs4 import BeautifulSoup
# import scipy.spatial.distance as sdist
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity#, paired_distances
from gensim.models import KeyedVectors
from flask import Flask, request
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


def mean_filtered(embed, doc):
    filtered = [x for x in doc.split() if x in embed.vocab]
    if len(filtered) > 0:
        return embed[filtered].mean(axis = 0)
    else:
        return np.zeros(embed.vector_size)


# sess = requests.Session()
# wapi_url = "https://en.wikipedia.org/w/api.php"

tok = Tokenizer()


app = Flask(__name__)
CORS(app)


@app.route('/')
@app.route('/index')
def index():
    return "It works!"


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

    origin_content = content['origin_content']
    target_content = content['target_content']

    origin_context_a = content['origin_context_a']
    origin_context_p = content['origin_context_p']

    origin_context = origin_context_p
    for origin_context_p_sentence in nltk.sent_tokenize(origin_context_p):
        if origin_context_a in origin_context_p_sentence:
            origin_context = origin_context_p_sentence.strip()

    origin_title_tokens = tok.load(origin_title).word_tokenize(lemmatize = True).word_tokens

    origin_sentence_htmls = []
    origin_sentence_tokens = []
    origin_context_sentence_ind = 0
    i = 0
    for origin_sentence_html in nltk.sent_tokenize(origin_content):
        origin_sentence_html = origin_sentence_html.strip()
        if origin_sentence_html == origin_context:
            origin_context_sentence_ind = i
        origin_sentence_soup = BeautifulSoup(origin_sentence_html, 'html5lib').body
        origin_sentence_htmls.append(origin_sentence_soup.decode_contents())
        origin_sentence_text = re.sub('\[.*?\]', '', origin_sentence_soup.get_text())
        origin_word_tokens = tok.load(origin_sentence_text).word_tokenize(lemmatize = True).word_tokens
        if len(origin_word_tokens) > 0:
            origin_sentence_tokens.append(' '.join(origin_word_tokens))
        i += 1

    target_sentence_htmls = []
    target_sentence_tokens = []
    for target_sentence_html in nltk.sent_tokenize(target_content):
        target_sentence_soup = BeautifulSoup(target_sentence_html, 'html5lib').body
        target_sentence_htmls.append(target_sentence_soup.decode_contents())
        target_sentence_text = re.sub('\[.*?\]', '', target_sentence_soup.get_text())
        target_word_tokens = tok.load(target_sentence_text).word_tokenize(lemmatize = True).word_tokens
        if len(target_word_tokens) > 0:
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

    origin_sentence_vectors = np.array([mean_filtered(embed, sent) for sent in origin_sentence_tokens])
    target_sentence_vectors = np.array([mean_filtered(embed, sent) for sent in target_sentence_tokens])

    top_match_indices = cosine_similarity(origin_sentence_vectors, target_sentence_vectors)[origin_context_sentence_ind].argsort()[::-1]

    n_top_matches = 5
    # sent_range = 2

    # hover_text = '\n'.join([' '.join(target_sentence_htmls[i - sent_range:i + sent_range + 1]) for i in top_match_indices[:n_top_matches]])
    # hover_text = '<br>---<br>'.join([target_sentence_htmls[i] for i in top_match_indices[:n_top_matches]])
    hover_text = '<br>'.join([target_sentence_htmls[i] for i in top_match_indices[:n_top_matches]])
    
    return str.encode(hover_text)


if __name__ == '__main__':
    embed = load_embedding(models_path + '/wiki2vec/en.model.kv')
    print("Model successfully loaded.")

    app.run(debug = True, host = '0.0.0.0', port = 5000)