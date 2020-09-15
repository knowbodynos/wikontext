import re
import numpy as np


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
