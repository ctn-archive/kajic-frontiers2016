from nengo import spa
import numpy as np
from numpy.linalg import svd


def svd_factorize(W, nr_dim):
    """
    Return a low-dimensional representation of a matrix symmetric matrix W.

    Parameters
    ----------
    W : (N, N) ndarray
        Symmetric input matrix.
    nrdim : int
        Desired dimensionality.

    Returns
    -------
    r_mat : (N, nr_dim) ndarray
        Matrix factor with np.dot(r_mat, r_mat.T) approximately equal to W.

    """
    u, s, _ = svd(W)
    ur = u[:, :nr_dim]
    sr = np.diag(np.sqrt(s[:nr_dim]))
    return np.dot(ur, sr)


def randomize(W, nr_dim):
    vocab = spa.Vocabulary(nr_dim)
    for i in range(len(W)):
        vocab.parse('V' + str(i))
    return vocab.vectors


def randomize_orthonormal(W, nr_dim):
    projection = np.random.randn(nr_dim, nr_dim)
    projection /= np.linalg.norm(projection, axis=1)[:, None]

    for i in range(1, nr_dim):
        projection[i] -= np.dot(projection[:i].T, np.dot(
            projection[:i], projection[i]))
        projection[i] /= np.linalg.norm(projection[i])
    return projection[:len(W), :]
