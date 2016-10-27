"""Methods for  saving and loading a set of vectors."""

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

import numpy as np


def from_assoc_matrix(assoc_matrix, dimred, n_dim):
    assoc_matrix = np.copy(assoc_matrix)
    np.fill_diagonal(assoc_matrix, 1.)
    reduced = dimred(assoc_matrix, n_dim)
    norms = np.linalg.norm(reduced, axis=1)
    norms = np.where(norms, norms, np.ones(len(norms)))
    reduced /= norms[:, None]
    return reduced


def save_pointers(path, name, pointers, id2word, word2id):
    """Save an association matrix.

    Parameters
    ----------
    path : str
        Output directory.
    name : str
        Filename without extension.
    pointers : (N, D) ndarray
        Array of N semantic pointers of dimensionality D.
    id2word: sequence/dict
        Mapping from index to word.
    word2id: dict
        Mapping from word to index.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    np.save(mat_file, pointers)
    with open(map_file, 'wb') as f:
        pickle.dump(id2word, f, protocol=2)
        pickle.dump(word2id, f, protocol=2)


def load_pointers(path, name):
    """Load semantic pointers.

    Parameters
    ----------
    path : str
        Input directory
    name : str
        Filename without extension.

    Returns:
    --------
    tuple
        (pointers, id2word, word2id) with the matrix of semantic pointers,
        mapping from matrix indices to words id2word, and mapping from words to
        matrix indices.
    """
    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    pointers = np.load(mat_file)
    with open(map_file, 'rb') as f:
        id2word = pickle.load(f)
        word2id = pickle.load(f)

    return pointers, id2word, word2id
