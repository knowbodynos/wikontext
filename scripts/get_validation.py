# Basic tools
import os
import sys
import re

# Data tools
import numpy as np
import pandas as pd
from scipy.stats import norm

# Scraping tools
import requests
from bs4 import BeautifulSoup

# Viz tools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# NLP tools
import nltk
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from sklearn.externals import joblib
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression

# Local
## Allow local relative imports
module_path = os.path.abspath('..')
include_path = os.path.join(module_path, 'include')
data_path = os.path.join(module_path, 'data')
models_path = os.path.join(module_path, 'models')
if include_path not in sys.path:
    sys.path.append(include_path)
from my_nlp import Tokenizer

def mean_filtered(embed, doc, vocab_weights = None):
    embed_filtered = [x for x in doc.split() if x in embed.vocab]
    if len(embed_filtered) > 0:
        if vocab_weights is None:
            return embed[embed_filtered].mean(axis = 0)
        else:
            weight_filtered = [x for x in embed_filtered if x in vocab_weights]
            if len(weight_filtered) > 0:
                weights = np.array([vocab_weights[x] for x in weight_filtered])
                return weights.dot(wiki2vec_embed[weight_filtered]) / weights.sum()
            else:
                return np.zeros(embed.vector_size, dtype = np.float32)
    else:
        return np.zeros(embed.vector_size, dtype = np.float32)

def logreg_sim(logreg, x, y):
    n_rows = x.shape[0]
    n_cols = y.shape[0]
    x_inds, y_inds = np.array(np.meshgrid(np.arange(n_rows), np.arange(n_cols))).reshape(2, -1)
    return logreg.predict_proba(np.concatenate([x[x_inds], y[y_inds], 1 - paired_distances(x[x_inds], y[y_inds]).reshape(-1, 1)], axis = 1))[:, 1].reshape((n_cols, n_rows)).T

def precision_recall(target_relevant_inds, target_selected_inds):
    n_selected = target_selected_inds.shape[1]
    n_relevant = len(target_relevant_inds)
    n_selected_relevant = intersect_vectorizer(target_relevant_inds, target_selected_inds)

    precision = n_selected_relevant / n_selected
    recall = n_selected_relevant / n_relevant
    return [precision, recall]

def populate_dataframe(sim, vectorizer, origin_title, target_title,
                       origin_sents, target_sents,
                       origin_relevant_inds, target_relevant_inds,
                       n_selected = 10, scale = 10):
    sim_mat = sim(*[vectorizer(x) for x in [origin_sents, target_sents]])
    sim_mat_T = sim_mat.T
    
    n_origin_relevent_inds = len(origin_relevant_inds)
    linked_titles = [np.array([x] * n_origin_relevent_inds) for x in [origin_title, target_title]]
    
    target_selected_inds_link = sim_mat[origin_relevant_inds].argsort(axis = 1)[:, :n_selected]
    target_selected_inds_wavg = weight_vectorizer(origin_sents, origin_relevant_inds, scale).dot(sim_mat).argsort(axis = 1)[:, :n_selected]
    target_selected_inds_summ = np.tile(np.arange(n_selected), (len(origin_relevant_inds), 1))
    target_selected_inds_rand = np.random.choice(np.arange(len(target_sents)), size = (n_origin_relevent_inds, n_selected))
    
    target_precision_recall = []
    for x in [target_selected_inds_link, target_selected_inds_wavg,
              target_selected_inds_summ, target_selected_inds_rand]:
        target_precision_recall.extend(precision_recall(target_relevant_inds, x))
    
    target_df = pd.DataFrame(linked_titles + target_precision_recall, index = ['origin_title', 'target_title',
                                                                               'link_precision', 'link_recall',
                                                                               'wavg_precision', 'wavg_recall',
                                                                               'summ_precision', 'summ_recall',
                                                                               'rand_precision', 'rand_recall']).T
    
    n_target_relevent_inds = len(target_relevant_inds)
    linked_titles = [np.array([x] * n_target_relevent_inds) for x in [target_title, origin_title]]
    
    origin_selected_inds_link = sim_mat_T[target_relevant_inds].argsort(axis = 1)[:, :n_selected]
    origin_selected_inds_wavg = weight_vectorizer(target_sents, target_relevant_inds, scale).dot(sim_mat_T).argsort(axis = 1)[:, :n_selected]
    origin_selected_inds_summ = np.tile(np.arange(n_selected), (len(target_relevant_inds), 1))
    origin_selected_inds_rand = np.random.choice(np.arange(len(origin_sents)), size = (len(target_relevant_inds), n_selected))    
    
    origin_precision_recall = []
    for x in [origin_selected_inds_link, origin_selected_inds_wavg,
              origin_selected_inds_summ, origin_selected_inds_rand]:
        origin_precision_recall.extend(precision_recall(origin_relevant_inds, x))
    
    origin_df = pd.DataFrame(linked_titles + origin_precision_recall, index = ['origin_title', 'target_title',
                                                                               'link_precision', 'link_recall',
                                                                               'wavg_precision', 'wavg_recall',
                                                                               'summ_precision', 'summ_recall',
                                                                               'rand_precision', 'rand_recall']).T
    
    return pd.concat([target_df, origin_df], axis = 0, ignore_index = True)

tok = Tokenizer()
sess = requests.Session()
wapi_url = "https://en.wikipedia.org/w/api.php"
tfidf_vectorizer = joblib.load(models_path + '/tfidf/enwiki-latest-all-tfidf_vectorizer.joblib')
wiki2vec_embed = KeyedVectors.load(models_path + '/wiki2vec/en.model.kv')
logreg = joblib.load(models_path + '/logreg.joblib')
wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x), signature = '()->(n)')
tfidf_wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x, vocab_weights = tfidf_vectorizer.vocabulary_), signature = '()->(n)')
weight_vectorizer = np.vectorize(lambda x, y, z: norm.pdf(range(x.size), loc = y, scale = z), signature = '(m),(),()->(n)')
intersect_vectorizer = np.vectorize(lambda x, y: np.intersect1d(x, y).size, signature = '(m),(n)->()')

n_bilinks = int(sys.argv[1])

bilinks_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-bilinks_sample.tsv', sep = '\t')

if os.path.exists(data_path + '/clickstream-enwiki-2018-08-validation.tsv'):
    valid_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t')
    start_bilink_ind = len(valid_df.groupby(by = ['origin_title', 'target_title'])) // 2
    valid_ind = valid_df.shape[0]
else:
    start_bilink_ind = 0
    valid_ind = 0

for bilink_ind in range(start_bilink_ind, start_bilink_ind + n_bilinks):
    origin_title, target_title = bilinks_df[['origin_title', 'target_title']].iloc[bilink_ind].tolist()

    origin_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': origin_title,
        'prop': "text",
        'contentformat': "text/plain",
        'format': "json"
    }

    try:
        origin_data = sess.get(url = wapi_url, params = origin_wapi_params).json()
        origin_title_norm = origin_data['parse']['title']
        origin_text = origin_data['parse']['text']['*']
    except KeyError:
        continue

    origin_soup = BeautifulSoup(origin_text, 'html5lib')

    target_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': target_title,
        'prop': "text",
        'format': "json"
    }

    try:
        target_data = sess.get(url = wapi_url, params = target_wapi_params).json()
        target_title_norm = target_data['parse']['title']
        target_text = target_data['parse']['text']['*']
    except KeyError:
        continue
    
    target_soup = BeautifulSoup(target_text, 'html5lib')

    origin_sents = []
    origin_relevant_inds = []
    sent_ind = 0
    for p in origin_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                sent_content = BeautifulSoup(sent, 'html5lib').body.get_text().replace(u'\xa0', u' ')
                sent_content = re.sub('\[.*?\]', '', sent_content)
                sent_content = ' '.join(tok.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                if len(sent_content) > 0:
                    if target_title_norm in sent:
                        origin_relevant_inds.append(sent_ind)
                    origin_sents.append(sent_content)
                    sent_ind += 1
    origin_sents = np.array(origin_sents)

    target_sents = []
    target_relevant_inds = []
    sent_ind = 0
    for p in target_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                sent_content = BeautifulSoup(sent, 'html5lib').body.get_text().replace(u'\xa0', u' ')
                sent_content = re.sub('\[.*?\]', '', sent_content)
                sent_content = ' '.join(tok.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                if len(sent_content) > 0:
                    if origin_title_norm in sent:
                        target_relevant_inds.append(sent_ind)
                    target_sents.append(sent_content)
                    sent_ind += 1
    target_sents = np.array(target_sents)

    if (origin_sents.size > 0) and (len(origin_relevant_inds) > 0) and (target_sents.size > 0) and (len(target_relevant_inds) > 0):
        cos_wiki2vec_df = populate_dataframe(cosine_similarity, wiki2vec_vectorizer, origin_title, target_title,
                                             origin_sents, target_sents,
                                             origin_relevant_inds, target_relevant_inds,
                                             n_selected = 10, scale = 10).drop(columns = ['origin_title', 'target_title'])

        cos_wiki2vec_df = cos_wiki2vec_df.rename(columns = dict([(x, 'cos_' + x) for x in cos_wiki2vec_df.columns]))

        logreg_wiki2vec_df = populate_dataframe(lambda x, y: logreg_sim(logreg, x, y), wiki2vec_vectorizer, origin_title, target_title,
                                                origin_sents, target_sents,
                                                origin_relevant_inds, target_relevant_inds,
                                                n_selected = 10, scale = 10).drop(columns = ['origin_title', 'target_title'])

        logreg_wiki2vec_df = logreg_wiki2vec_df.rename(columns = dict([(x, 'logreg_' + x) for x in logreg_wiki2vec_df.columns]))

        cos_tfidf_wiki2vec_df = populate_dataframe(cosine_similarity, tfidf_wiki2vec_vectorizer, origin_title, target_title,
                                                   origin_sents, target_sents,
                                                   origin_relevant_inds, target_relevant_inds,
                                                   n_selected = 10, scale = 10).drop(columns = ['origin_title', 'target_title'])

        cos_tfidf_wiki2vec_df = cos_tfidf_wiki2vec_df.rename(columns = dict([(x, 'cos_tfidf_' + x) for x in cos_tfidf_wiki2vec_df.columns]))

        logreg_tfidf_wiki2vec_df = populate_dataframe(lambda x, y: logreg_sim(logreg, x, y), tfidf_wiki2vec_vectorizer, origin_title, target_title,
                                                      origin_sents, target_sents,
                                                      origin_relevant_inds, target_relevant_inds,
                                                      n_selected = 10, scale = 10).drop(columns = ['origin_title', 'target_title'])

        logreg_tfidf_wiki2vec_df = logreg_tfidf_wiki2vec_df.rename(columns = dict([(x, 'logreg_tfidf_' + x) for x in logreg_tfidf_wiki2vec_df.columns]))

        valid_df = pd.concat([cos_wiki2vec_df, logreg_wiki2vec_df, cos_tfidf_wiki2vec_df, logreg_tfidf_wiki2vec_df], axis = 1).dropna().reset_index(drop = True)
    
        if os.path.exists(data_path + '/clickstream-enwiki-2018-08-validation.tsv'):
            valid_df.to_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t', mode = 'a', header = False, index = False)
        else:
            valid_df.to_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t', index = False)

        valid_ind += valid_df.shape[0]

        if valid_df.shape[0] > 0:
            print("Title Bilink: {}, Validation Point: {}".format(bilink_ind, valid_ind))