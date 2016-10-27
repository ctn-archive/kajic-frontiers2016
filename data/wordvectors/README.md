#### Semantic pointers

This directory contains vector representations of words generated using
different methods.

The naming convention for files in this directory is
`{data}_{method}_{nwords}w_{ndim}d.{npz,pkl}`. `{data}` is the association data
from which the vectors were obtained, `{method}` is the method to obtain the
vectors, `{nwords}` is the number of words/vectors (= number of rows in the
matrix), `{ndim}` is the dimensionality of the vectors (= number columns in the
matrix). The `.npz` file stores the matrix with the word vectors, the `.pkl`
file stores the mappings from matrix indices to words (ind2word) and words to
indices (word2ind).
