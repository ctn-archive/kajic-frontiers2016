from __future__ import print_function

import numpy as np
import os
from sparat.data_processing.generate_association_matrix import load_assoc_mat


assoc_name = 'freeassoc_asymmetric'
assoc_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
    'associationmatrices')
assoc, id2word, word2id = load_assoc_mat(assoc_dir, assoc_name)

def show_associates(word, assoc, word2id, id2word, th=0.):
    """
    For `assoc`, a given matrix of association strengths, print associates and
    their strengths for a given word `word`.
    """

    print('Associates of:', word)
    word = word.upper()
    assoc_idx = np.argsort(assoc[word2id[word]])[::-1]
    for idx in assoc_idx:
        w = assoc[word2id[word], idx]
        if w > th:
            print(id2word[idx], w)

def print_sims(sims, i2w, th=0.1):
    """
    For a given list of similarity values, print the words with the similarity value above
    th.
    """
    for i in np.argsort(sims)[::-1]:
        if sims[i] > th:
            print(i2w[i], sims[i])

