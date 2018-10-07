# Basic tools
import os
import sys
import re
# from time import sleep
from random import randint, sample

# Data tools
import numpy as np
import pandas as pd

# Scraping tools
import requests
from bs4 import BeautifulSoup

# NLP tools
import nltk

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

n_samples = int(sys.argv[1])

titles_df = pd.read_csv(data_path + '/enwiki-latest-all-titles.tsv', sep = '\t')

titles = titles_df[titles_df.page_namespace == 0].sample(frac = 1).page_title

with open(data_path + '/enwiki-latest-all-titles-sentence_samples.tsv', 'w') as sample_stream:
    print("sentence", file = sample_stream)
    title_ind = 0
    sent_ind = 0
    sample_ind = 0
    prev_sample_ind = 0
    while (title_ind < titles.size) and (sample_ind < n_samples):
        wapi_params = {
            'action': "parse",
            'maxlag': 5,
            'page': titles[title_ind],
            'prop': "text",
            'contentformat': "text/plain",
            'format': "json"
        }

        try:
            data = sess.get(url = wapi_url, params = wapi_params).json()
            text = data['parse']['text']['*']
        except KeyError:
            title_ind += 1
            continue

        if not '<div class="redirectMsg">' in text:
            sent_list = []
            soup = BeautifulSoup(text, 'html5lib')
            for p in soup.find_all('p'):
                for p_split in p.decode_contents().split('\n'):
                    for sent in nltk.sent_tokenize(p_split):
                        sent_content = BeautifulSoup(sent, 'html5lib').body.get_text().replace(u'\xa0', u' ')
                        sent_content = re.sub('\[.*?\]', '', sent_content)
                        sent_content = ' '.join(tok.load(sent_content).word_tokenize(lemmatize = True).word_tokens)
                        sent_list.append(sent_content)
                        sent_ind += 1
            sent_quota = randint(0, len(sent_list))
            print('\n'.join(sample(sent_list, sent_quota)), file = sample_stream)
            sample_ind += sent_quota
            if (sample_ind - prev_sample_ind) >= 500:
                print("Title: {}, Sentence: {}, Sample: {}".format(title_ind, sent_ind, sample_ind))
                prev_sample_ind = sample_ind
        title_ind += 1