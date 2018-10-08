# Basic tools
import os
import sys
import re

# Data tools
import numpy as np
import pandas as pd

# Scraping tools
import requests
from bs4 import BeautifulSoup

# NLP tools
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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

def sentence_lists(origin_title, target_title, session = requests.Session(), tokenizer = Tokenizer()):
    wapi_url = "https://en.wikipedia.org/w/api.php"
    
    origin_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': origin_title,
        'prop': "text",
        'contentformat': "text/plain",
        'format': "json"
    }

    try:
        origin_data = session.get(url = wapi_url, params = origin_wapi_params).json()
        origin_title_norm = origin_data['parse']['title']
        origin_text = origin_data['parse']['text']['*']
    except KeyError:
        return None

    origin_soup = BeautifulSoup(origin_text, 'html5lib')

    target_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': target_title,
        'prop': "text",
        'format': "json"
    }

    try:
        target_data = session.get(url = wapi_url, params = target_wapi_params).json()
        target_title_norm = target_data['parse']['title']
        target_text = target_data['parse']['text']['*']
    except KeyError:
        return None
    
    target_soup = BeautifulSoup(target_text, 'html5lib')

    origin_sents = []
    origin_relevant_inds = []
    sent_ind = 0
    for p in origin_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                if sent != b'.':
                    sent_soup = BeautifulSoup(sent, 'html5lib').body
                    sent_content = sent_soup.get_text().replace(u'\xa0', u' ')
                    sent_content = re.sub('\[.*?\]', '', sent_content)
                    sent_content = ' '.join(tokenizer.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                    if len(sent_content) > 1:
                        for a in sent_soup.find_all('a'):
                            if (('href' in a.attrs) and (a.attrs['href'] == "/wiki/" + target_title)) or (target_title_norm in sent):
                                origin_relevant_inds.append(sent_ind)
                                break
                        origin_sents.append(sent_content)
                        sent_ind += 1
    origin_sents = np.array(origin_sents)

    target_sents = []
    target_relevant_inds = []
    sent_ind = 0
    for p in target_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                if sent != b'.':
                    sent_soup = BeautifulSoup(sent, 'html5lib').body
                    sent_content = sent_soup.get_text().replace(u'\xa0', u' ')
                    sent_content = re.sub('\[.*?\]', '', sent_content)
                    sent_content = ' '.join(tokenizer.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                    if len(sent_content) > 1:
                        for a in sent_soup.find_all('a'):
                            if (('href' in a.attrs) and (a.attrs['href'] == "/wiki/" + origin_title)) or (origin_title_norm in sent):
                                target_relevant_inds.append(sent_ind)
                                break
                        target_sents.append(sent_content)
                        sent_ind += 1
    target_sents = np.array(target_sents)
    
    return [origin_sents, target_sents], [origin_relevant_inds, target_relevant_inds]

def get_df(sentences, indices, rand_indices, n_selected, v_cols, vect, dist):
    origin_sents, target_sents = sentences
    origin_inds, target_inds = indices
    target_rand_inds, origin_rand_inds = rand_indices
    dist_mat = dist(*[vect(x) for x in sentences])
    
    df = pd.DataFrame([], columns = v_cols)

    for i, rand_j in zip(origin_inds, target_rand_inds):
        j = dist_mat[i].argmin()
        linked = dist_mat[i, j]
        baseline = dist_mat[i, 0]
#         random = np.random.choice(dist_mat[i])
        random = dist_mat[i, rand_j]
        dist_row = dist_mat[i]
        dist_inds_selected = dist_row.argsort()
        n_relevant_selected = np.intersect1d(dist_inds_selected[:n_selected], target_inds).size
        avg_prec = sum(np.intersect1d(dist_inds_selected[:i + 1], target_inds).size / (i + 1) for i in range(dist_inds_selected.size) if dist_inds_selected[i] in target_inds) / len(target_inds)
        temp_df = pd.DataFrame(dict(zip(v_cols, [target_sents[j], target_sents[0], target_sents[rand_j], linked, baseline, random, n_relevant_selected, avg_prec])), index = [0])
        df = df.append(temp_df, ignore_index = True)

    for rand_i, j in zip(origin_rand_inds, target_inds):
        i = dist_mat[:, j].argmin()
        linked = dist_mat[i, j]
        baseline = dist_mat[0, j]
#         random = np.random.choice(dist_mat[:, j])
        random = dist_mat[rand_i, j]
        dist_col = dist_mat[:, j]
        dist_inds_selected = dist_col.argsort()
        n_relevant_selected = np.intersect1d(dist_inds_selected[:n_selected], origin_inds).size
        avg_prec = sum(np.intersect1d(dist_inds_selected[:i + 1], origin_inds).size / (i + 1) for i in range(dist_inds_selected.size) if dist_inds_selected[i] in origin_inds) / len(origin_inds)
        temp_df = pd.DataFrame(dict(zip(v_cols, [origin_sents[i], origin_sents[0], origin_sents[rand_i], linked, baseline, random, n_relevant_selected, avg_prec])), index = [0])
        df = df.append(temp_df, ignore_index = True)
    
    return df

tok = Tokenizer()
sess = requests.Session()

wiki2vec_embed = KeyedVectors.load(models_path + '/wiki2vec/en.model.kv')
wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x), signature = '()->(n)')
wiki2vec_tfidf_embed = joblib.load(models_path + '/tfidf/enwiki-latest-all-wiki2vec_tfidf_embed.joblib')
wiki2vec_idf = dict(zip(wiki2vec_tfidf_embed.get_feature_names(), wiki2vec_tfidf_embed.idf_))
tfidf_wiki2vec_vectorizer = np.vectorize(lambda x: mean_filtered(wiki2vec_embed, x, vocab_weights = wiki2vec_idf), signature = '()->(n)')

glove_embed = KeyedVectors.load_word2vec_format(models_path + '/glove/word2vec.glove.6B.300d.txt')
glove_vectorizer = np.vectorize(lambda x: mean_filtered(glove_embed, x), signature = '()->(n)')
glove_tfidf_embed = joblib.load(models_path + '/tfidf/enwiki-latest-all-glove_tfidf_embed.joblib')
glove_idf = dict(zip(glove_tfidf_embed.get_feature_names(), glove_tfidf_embed.idf_))
tfidf_glove_vectorizer = np.vectorize(lambda x: mean_filtered(glove_embed, x, vocab_weights = glove_idf), signature = '()->(n)')

logreg_wiki2vec_cos = joblib.load(models_path + '/logreg/logreg_wiki2vec_cos.joblib')
logreg_wiki2vec_euc = joblib.load(models_path + '/logreg/logreg_wiki2vec_euc.joblib')
logreg_tfidf_wiki2vec_cos = joblib.load(models_path + '/logreg/logreg_tfidf_wiki2vec_cos.joblib')
logreg_tfidf_wiki2vec_euc = joblib.load(models_path + '/logreg/logreg_tfidf_wiki2vec_euc.joblib')

logreg_glove_cos = joblib.load(models_path + '/logreg/logreg_glove_cos.joblib')
logreg_glove_euc = joblib.load(models_path + '/logreg/logreg_glove_euc.joblib')
logreg_tfidf_glove_cos = joblib.load(models_path + '/logreg/logreg_tfidf_glove_cos.joblib')
logreg_tfidf_glove_euc = joblib.load(models_path + '/logreg/logreg_tfidf_glove_euc.joblib')

n_bilinks = int(sys.argv[1])
n_selected = int(sys.argv[2])

bilinks_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-bilinks_sample.tsv', sep = '\t')

if os.path.exists(data_path + '/clickstream-enwiki-2018-08-validation.tsv'):
    valid_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t')
    inds_forward = bilinks_df[(bilinks_df['origin_title'] == valid_df.iloc[-1]['origin_title']) & \
                              (bilinks_df['target_title'] == valid_df.iloc[-1]['target_title'])].index.tolist()
    inds_backward = bilinks_df[(bilinks_df['target_title'] == valid_df.iloc[-1]['origin_title']) & \
                               (bilinks_df['origin_title'] == valid_df.iloc[-1]['target_title'])].index.tolist()
    start_bilink_ind = min(inds_forward + inds_backward) + 1
    valid_ind = valid_df.shape[0]
else:
    start_bilink_ind = 0
    valid_ind = 0

for bilink_ind in range(start_bilink_ind, start_bilink_ind + n_bilinks):
    origin_title, target_title = bilinks_df[['origin_title', 'target_title']].iloc[bilink_ind].tolist()
    result = sentence_lists(origin_title, target_title, session = sess, tokenizer = tok)
    if not result is None:
        sentences, indices = result
        if all([x.size > 0 for x in sentences]) and all([len(x) > 0 for x in indices]):
            origin_sents, target_sents = sentences
            origin_inds, target_inds = indices
            rand_indices = [np.random.choice(range(sentences[-1 - i].size), size = len(indices[i])) for i in range(len(indices))]

            v_cols = ['origin_title', 'target_title', 'origin_sent', 'n_relevant', 'n_selected']
            valid_df = pd.DataFrame([], columns = v_cols)

            for i in origin_inds:
                temp_df = pd.DataFrame(dict(zip(v_cols, [origin_title, target_title, origin_sents[i], len(target_inds), n_selected])), index = [0])
                valid_df = valid_df.append(temp_df, ignore_index = True)

            for j in target_inds:
                temp_df = pd.DataFrame(dict(zip(v_cols, [target_title, origin_title, target_sents[j], len(origin_inds), n_selected])), index = [0])
                valid_df = valid_df.append(temp_df, ignore_index = True)

            v_cols = ['target_sent_linked', 'target_sent_baseline', 'target_sent_random', 'dist_linked', 'dist_baseline', 'dist_random', 'n_relevant_selected', 'avg_prec']
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_wiki2vec_cos' for x in v_cols], wiki2vec_vectorizer, lambda x, y: 1 - cosine_similarity(x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_wiki2vec_euc' for x in v_cols], wiki2vec_vectorizer, euclidean_distances)], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_wiki2vec_logreg_cos' for x in v_cols], wiki2vec_vectorizer, lambda x, y: logreg_distance(logreg_wiki2vec_cos, x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_wiki2vec_logreg_euc' for x in v_cols], wiki2vec_vectorizer, lambda x, y: logreg_distance(logreg_wiki2vec_euc, x, y))], axis = 1)

            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_wiki2vec_cos' for x in v_cols], tfidf_wiki2vec_vectorizer, lambda x, y: 1 - cosine_similarity(x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_wiki2vec_euc' for x in v_cols], tfidf_wiki2vec_vectorizer, euclidean_distances)], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_wiki2vec_logreg_cos' for x in v_cols], tfidf_wiki2vec_vectorizer, lambda x, y: logreg_distance(logreg_tfidf_wiki2vec_cos, x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_wiki2vec_logreg_euc' for x in v_cols], tfidf_wiki2vec_vectorizer, lambda x, y: logreg_distance(logreg_tfidf_wiki2vec_euc, x, y))], axis = 1)

            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_glove_cos' for x in v_cols], glove_vectorizer, lambda x, y: 1 - cosine_similarity(x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_glove_euc' for x in v_cols], glove_vectorizer, euclidean_distances)], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_glove_logreg_cos' for x in v_cols], glove_vectorizer, lambda x, y: logreg_distance(logreg_glove_cos, x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_glove_logreg_euc' for x in v_cols], glove_vectorizer, lambda x, y: logreg_distance(logreg_glove_euc, x, y))], axis = 1)

            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_glove_cos' for x in v_cols], tfidf_glove_vectorizer, lambda x, y: 1 - cosine_similarity(x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_glove_euc' for x in v_cols], tfidf_glove_vectorizer, euclidean_distances)], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_glove_logreg_cos' for x in v_cols], tfidf_glove_vectorizer, lambda x, y: logreg_distance(logreg_tfidf_glove_cos, x, y))], axis = 1)
            valid_df = pd.concat([valid_df, get_df(sentences, indices, rand_indices, n_selected, [x + '_tfidf_glove_logreg_euc' for x in v_cols], tfidf_glove_vectorizer, lambda x, y: logreg_distance(logreg_tfidf_glove_euc, x, y))], axis = 1)

            if os.path.exists(data_path + '/clickstream-enwiki-2018-08-validation.tsv'):
                valid_df.to_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t', mode = 'a', header = False, index = False)
            else:
                valid_df.to_csv(data_path + '/clickstream-enwiki-2018-08-validation.tsv', sep = '\t', index = False)

            valid_ind += valid_df.shape[0]

            if valid_df.shape[0] > 0:
                print("Title Bilink: {}, Validation Point: {}".format(bilink_ind, valid_ind))