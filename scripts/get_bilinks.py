# Basic tools
import os
import sys

# Data tools
import numpy as np
import pandas as pd

# Local
## Allow local relative imports
module_path = os.path.abspath('..')
include_path = os.path.join(module_path, 'include')
data_path = os.path.join(module_path, 'data')
if include_path not in sys.path:
    sys.path.append(include_path)

# From https://dumps.wikimedia.org/other/clickstream
# Removed spiders and bots by filtering requests for articles by a small number of clients hundreds of times per minute within some time window

clickstream_df = pd.read_csv(data_path + '/clickstream-enwiki-2018-08.tsv', sep = '\t', names = ['origin_title', 'target_title', 'type', 'n_clicks'])
clickstream_df = clickstream_df.loc[df.type == 'link'].drop(columns = ['type']).sort_values(by = ['n'], ascending = False).reset_index(drop = True)

forward_df = clickstream_df.rename(columns = {'n_clicks': 'n_clicks_forward'})
backward_df = clickstream_df.rename(columns = {'target_title': 'origin_title', 'origin_title': 'target_title', 'n_clicks': 'n_clicks_backward'})
symm_df = pd.merge(forward_df, backward_df, on = ['origin_title', 'target_title'])
sort_df = np.sort(symm_df[['origin_title', 'target_title']].values.astype(str), axis = 1)
drop_df = pd.DataFrame(sort_df).drop_duplicates()
bilinks_df = symm_df.loc[drop_df.index].sort_values(by = ['n_clicks_forward'], ascending = False)

bilinks_df.to_csv(data_path + '/clickstream-enwiki-2018-08-bilinks.tsv', sep = '\t', index = False)