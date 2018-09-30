# Basic tools
import os
import sys
import re
# from time import sleep

# Data tools
import numpy as np
import pandas as pd
import scipy.spatial.distance as sdist
from scipy.stats import norm

# Scraping tools
import requests
from bs4 import BeautifulSoup

# Viz tools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# NLP tools
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Local
## Allow local relative imports
module_path = os.path.abspath('..')
include_path = os.path.join(module_path, 'include')
data_path = os.path.join(module_path, 'data')
if include_path not in sys.path:
    sys.path.append(include_path)
from my_nlp import Tokenizer

tok = Tokenizer()
sess = requests.Session()
wapi_url = "https://en.wikipedia.org/w/api.php"

n_bilinks = int(sys.argv[1])

bilinks_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-bilinks.tsv', sep = '\t')

count_df_cols = ['origin_p_count', 'origin_newline_count', 'origin_sent_count', 'n_links_forward',
                 'target_p_count', 'target_newline_count', 'target_sent_count', 'n_links_backward']

for count_df_col in count_df_cols:
    bilinks_df[count_df_col] = 0

pos_df_cols = ['origin_p_pos', 'origin_newline_pos', 'origin_sent_pos', 'origin_sent', 'origin_first_sent',
               'target_p_pos', 'target_newline_pos', 'target_sent_pos', 'target_sent', 'target_first_sent', 'match']

if os.path.exists(data_path + '/clickstream-enwiki-2018-08-sentences.tsv'):
    sent_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08-sentences.tsv', sep = '\t')
    start_bilink_ind = len(sent_df.groupby(by = ['origin_title', 'target_title']))
    sent_ind = sent_df.shape[0]
else:
    sent_df = pd.DataFrame(columns = bilinks_df.columns)

    for pos_df_col in pos_df_cols:
        sent_df[pos_df_col] = 0

    sent_df.to_csv(data_path + '/clickstream-enwiki-2018-08-sentences.tsv', sep = '\t', index = False)
    start_bilink_ind = 0
    sent_ind = 0

for bilink_ind in range(start_bilink_ind, start_bilink_ind + n_bilinks):
    sent_df = pd.DataFrame(columns = bilinks_df.columns)

    for pos_df_col in pos_df_cols:
        sent_df[pos_df_col] = 0

    origin_title, target_title = bilinks_df[['origin_title', 'target_title']].iloc[bilink_ind].tolist()

    origin_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': origin_title,
        'prop': "text",
        'contentformat': "text/plain",
        'format': "json"
    }

    # while True:
    try:
        origin_data = sess.get(url = wapi_url, params = origin_wapi_params).json()
        origin_title_norm = origin_data['parse']['title']
        origin_text = origin_data['parse']['text']['*']
    except KeyError:
        continue
        #     sleep(1)
        # else:
        #     break
    if isinstance(origin_text, bytes):
        origin_text = origin_text.decode()
    origin_soup = BeautifulSoup(origin_text, 'html5lib')

    target_wapi_params = {
        'action': "parse",
        'maxlag': 5,
        'page': target_title,
        'prop': "text",
        'format': "json"
    }

    # while True:
    try:
        target_data = sess.get(url = wapi_url, params = target_wapi_params).json()
        target_title_norm = target_data['parse']['title']
        target_text = target_data['parse']['text']['*']
    except KeyError:
        continue
        #     sleep(1)
        # else:
        #     break
    if isinstance(target_text, bytes):
        target_text = target_text.decode()
    target_soup = BeautifulSoup(target_text, 'html5lib')

    origin_ref_sents = []
    origin_temp_sents = []
    p_pos = 0
    newline_pos = 0
    sent_pos = 0
    for p in origin_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                sent_content = BeautifulSoup(sent, 'html5lib').body.get_text()#.decode("string_escape")
                sent_content = re.sub('\[.*?\]', '', sent_content)
                sent_content = ' '.join(tok.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                bilinks_df.loc[bilink_ind, 'origin_sent_count'] += 1
                if sent_pos == 0:
                    origin_first_sent = sent_content
                sent_pos += 1
                if target_title_norm in sent:
                    origin_ref_sents.append([p_pos, newline_pos, sent_pos, sent_content])
                    bilinks_df.loc[bilink_ind, 'n_links_forward'] += 1
                else:
                    origin_temp_sents.append([p_pos, newline_pos, sent_pos, sent_content])
            bilinks_df.loc[bilink_ind, 'origin_newline_count'] += 1
            newline_pos += 1
        bilinks_df.loc[bilink_ind, 'origin_p_count'] += 1
        p_pos += 1

    target_ref_sents = []
    target_temp_sents = []
    p_pos = 0
    newline_pos = 0
    sent_pos = 0
    for p in target_soup.find_all('p'):
        for p_split in p.decode_contents().split('\n'):
            for sent in nltk.sent_tokenize(p_split):
                sent_content = BeautifulSoup(sent, 'html5lib').body.get_text()#.decode("string_escape")
                sent_content = re.sub('\[.*?\]', '', sent_content)
                sent_content = ' '.join(tok.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                bilinks_df.loc[bilink_ind, 'target_sent_count'] += 1
                if sent_pos == 0:
                    target_first_sent = sent_content
                sent_pos += 1
                if origin_title_norm in sent:
                    target_ref_sents.append([p_pos, newline_pos, sent_pos, sent_content])
                    bilinks_df.loc[bilink_ind, 'n_links_backward'] += 1
                else:
                    target_temp_sents.append([p_pos, newline_pos, sent_pos, sent_content])
            bilinks_df.loc[bilink_ind, 'target_newline_count'] += 1
            newline_pos += 1
        bilinks_df.loc[bilink_ind, 'target_p_count'] += 1
        p_pos += 1

    for o in origin_ref_sents:
        for t in target_ref_sents:
            temp_df = pd.DataFrame(dict(zip(bilinks_df.columns.tolist() + pos_df_cols, bilinks_df.iloc[bilink_ind].tolist() + o + [origin_first_sent] + t + [target_first_sent] + [1])), index = [0])
            if temp_df.isna().values.any():
                print(temp_df)
            sent_df = sent_df.append(temp_df, sort = False, ignore_index = True).reset_index(drop = True)
    
    if (len(origin_temp_sents) > len(origin_ref_sents)) and (len(target_temp_sents) > len(target_ref_sents)):
        origin_rand_sents = map(lambda i: origin_temp_sents[i], np.random.choice(np.arange(len(origin_temp_sents)), replace = False, size = len(origin_ref_sents)).tolist())
        target_rand_sents = map(lambda i: target_temp_sents[i], np.random.choice(np.arange(len(target_temp_sents)), replace = False, size = len(target_ref_sents)).tolist())
        for o in origin_rand_sents:
            for t in target_rand_sents:
                temp_df = pd.DataFrame(dict(zip(bilinks_df.columns.tolist() + pos_df_cols, bilinks_df.iloc[bilink_ind].tolist() + o + [origin_first_sent] + t + [target_first_sent] + [0])), index = [0])
                if temp_df.isna().values.any():
                    print(temp_df)
                sent_df = sent_df.append(temp_df, sort = False, ignore_index = True).reset_index(drop = True)
    
    sent_df = sent_df.dropna().reset_index(drop = True)
    sent_df.dropna().to_csv(data_path + '/clickstream-enwiki-2018-08-sentences.tsv', sep = '\t', mode = 'a', header = False, index = False)

    sent_ind += sent_df.shape[0]

    if sent_df.shape[0] > 0:
        print("Title Bilink: {}, Sentence Bilink: {}".format(bilink_ind, sent_ind))