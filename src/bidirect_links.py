import os
import sys
import re
import numpy as np
import pandas as pd
import pickle
import requests
import mwparserfromhell
import scipy.spatial.distance as sdist
from itertools import zip_longest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

module_path = os.path.abspath('..')
include_path = os.path.join(module_path, 'include')
if include_path not in sys.path:
    sys.path.append(include_path)

from my_nlp import Tokenizer

def grouper(iterable, n, fillvalue = None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue = fillvalue)

titles = []
with open(module_path + "/data/titles-sorted.txt", "r") as titles_stream:
    for title in titles_stream:
        titles.append(title.rstrip('\n'))

links_dict = {}
with open(module_path + "/data/links-simple-sorted.txt", "r") as links_stream:
    for links in links_stream:
        origin_str, targets_str = links.rstrip('\n').split(': ')
        origin = int(origin_str)
        links_dict[origin] = []
        for target in targets_str.split():
            target = int(target)
            links_dict[origin].append(target)

for origin, targets in list(links_dict.items()):
    for target in list(targets):
        if not (target in links_dict and origin in links_dict[target]):
            links_dict[origin].remove(target)
    if len(links_dict[origin]) == 0:
        del links_dict[origin]

with open(module_path + '/data/links.pickle', 'wb') as handle:
    pickle.dump(links_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open(module_path + '/data/titles.pickle', 'wb') as handle:
    pickle.dump(titles, handle, protocol = pickle.HIGHEST_PROTOCOL)