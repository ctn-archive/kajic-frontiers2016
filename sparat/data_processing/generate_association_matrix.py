from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
import itertools
import os
import os.path
import string

import numpy as np


def gen_asymmetric(words, association_database, diag=1.):
    """Generates an asymmetric association matrix from cues (rows) to targets
    (columns).

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    id2word = list(words)
    word2id = {w: i for i, w in enumerate(id2word)}
    strength_mat = np.zeros((len(words), len(words)))
    for cue, target, strength in association_database:
        strength_mat[word2id[cue], word2id[target]] += strength
    np.fill_diagonal(strength_mat, diag)
    assert np.all(strength_mat <= 1.)
    return strength_mat, id2word, word2id


def gen_symmetric(words, association_database, diag=1.):
    """Generates a symmetric association matrix from cues (rows) to targets
    (columns). This is done by using the asymmetric association matrix W and
    calculating (W + W.T)/2.

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    return (asymetric + asymetric.T) / 2., id2word, word2id


def gen_symmetric2(words, association_database, diag=1.):
    """Generates a symmetric association matrix from cues (rows) to targets
    (columns). This symmetric matrix also includes second measure for
    associative strength:

    S_ij(2) = S_ij(1) + sum_k S_ik(1)*S_kj(1)

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    symmetric = (asymetric + asymetric.T) / 2.
    symmetric2 = symmetric + np.dot(symmetric, symmetric)

    return symmetric2, id2word, word2id


def gen_undir_asymmetric(words, association_database, diag=1.):
    """Generates an undirected asymmetric association matrix from cues (rows) to
    targets (columns). All non-zero association strengths between a cue and
    a target are set to 1.

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymmetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    undir_asymmetric = np.where(asymmetric > 0., 1., 0.)

    return undir_asymmetric, id2word, word2id


def gen_undir_symmetric(words, association_database, diag=1.):
    """Generates an undirected symmetric association matrix from cues (rows) to
    targets (columns). All non-zero association strengths between a cue and
    a target are set to 1.

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymmetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    symmetric = (asymmetric + asymmetric.T) / 2.
    undir_symmetric = np.where(symmetric > 0., 1., 0.)

    return undir_symmetric, id2word, word2id


def gen_google_symmetric(words, association_database, diag=1.):
    """Generates a symmetric association matrix by adding free association
    symmetric matrix and google bi-grams matrix. Replaces all non-zero
    associations with ones.

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymmetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    symmetric = (asymmetric + asymmetric.T) / 2.

    bigram_path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data',
        'associationmatrices')

    bigrams, id2word, word2id = load_assoc_mat(
        bigram_path, 'google_symmetric')

    both = bigrams + symmetric
    both = np.where(both > 0., 1., 0.)

    return both, id2word, word2id


def tr_binary(mat):
    return mat > 0


def tr_normalize(mat):
    norms = np.linalg.norm(mat, axis=1)
    norms = np.where(norms, norms, np.ones(len(norms)))
    return mat / norms[:, None]


def tr_symmetrify(mat):
    return (mat + mat.T) / 2.


def tr_add(mat1, mat2):
    return mat1 + mat2


def tr_multiply(mat1, mat2):
    return np.dot(mat1, mat2)


def tr_ppmi(mat):
    denominator = np.asarray(
        np.sum(mat, axis=0)[None, :] * np.sum(mat, axis=1)[:, None],
        dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.log(mat * np.sum(mat) / denominator)
        mat[~np.isfinite(mat)] = 0
        mat[mat < 0] = 0
    return mat


def tr_log(mat):
    epsilon = 0.001
    return np.log(mat + epsilon)


def gen_1grams(words):
    from joblib import Parallel, delayed

    id2word = list(words)
    word2id = {w: i for i, w in enumerate(id2word)}
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data',
        'associationmatrices', 'google-1grams')
    strength_mat = np.memmap(
        path, shape=(len(words), len(words)), mode='w+',
        dtype='uint32')
    strength_mat.fill(0.)
    Parallel(n_jobs=5)(
        delayed(process_1gram_file)(c, word2id, strength_mat)
        for c in string.ascii_lowercase)
    return strength_mat, id2word, word2id


def process_1gram_file(c, word2id, output):
    template = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw',
        'google', 'googlebooks-eng-all-1gram-20120701-{c}.gz')
    filename = template.format(c=c)
    print(filename)

    with gzip.open(filename, 'rt') as f:
        for line in f:
            ngram, year, count, _ = line.split('\t')
            if year != '2008':
                continue
            ngram = ngram.upper()
            for i in range(1, len(ngram) - 1):
                words = (ngram[:i], ngram[i:])
                if any(w not in word2id for w in words):
                    continue
                output[word2id[words[0]], word2id[words[1]]] = int(count)


def gen_bigrams(words):
    from joblib import Parallel, delayed

    id2word = list(words)
    word2id = {w: i for i, w in enumerate(id2word)}
    cs = itertools.product(string.ascii_lowercase, string.ascii_lowercase)

    path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data',
        'associationmatrices', 'google-bigrams')
    strength_mat = np.memmap(
        path, shape=(len(words), len(words)), mode='w+',
        dtype='uint32')
    strength_mat.fill(0.)
    Parallel(n_jobs=5)(
        delayed(process_bigram_file)(c1, c2, word2id, strength_mat)
        for c1, c2 in cs)
    return strength_mat, id2word, word2id


def process_bigram_file(c1, c2, word2id, output):
    template = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw',
        'google', 'googlebooks-eng-all-2gram-20120701-{c}.gz')
    filename = template.format(c=c1 + c2)
    print(filename)

    with gzip.open(filename, 'rt') as f:
        for line in f:
            ngram, year, count, _ = line.split('\t')
            if year != '2008':
                continue
            words = ngram.upper().split(' ')
            if any(w not in word2id for w in words):
                continue
            output[word2id[words[0]], word2id[words[1]]] = int(count)


def print_stats(mat):
    assoc_counts = np.sum(mat > 0., axis=1)
    print(
        "Average number of associates per word:", np.mean(assoc_counts),
        np.std(assoc_counts))


def save_assoc_mat(path, name, strength_mat, id2word, word2id):
    """Save an association matrix.

    Parameters
    ----------
    path : str
        Output directory.
    name : str
        Filename without extension.
    strength_mat : ndarray
        Association matrix.
    id2word: sequence/dict
        Mapping from index to word.
    word2id: dict
        Mapping from word to index.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    np.save(mat_file, strength_mat)
    with open(map_file, 'wb') as f:
        pickle.dump(id2word, f, protocol=2)
        pickle.dump(word2id, f, protocol=2)


def load_assoc_mat(path, name):
    """Load an association matrix.

    Parameters
    ----------
    path : str
        Input directory
    name : str
        Filename without extension.

    Returns:
    --------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    strength_mat = np.load(mat_file)
    with open(map_file, 'rb') as f:
        id2word = pickle.load(f)
        word2id = pickle.load(f)

    return strength_mat, id2word, word2id
