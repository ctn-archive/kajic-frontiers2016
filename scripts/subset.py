#!/usr/bin/env python

import os
import os.path
import argparse

import numpy as np

from sparat.data_processing import generate_association_matrix


parser = argparse.ArgumentParser(
    description="Create a subset association matrix.")
parser.add_argument('input', nargs=1)
parser.add_argument('output', nargs=1)
parser.add_argument('hops', nargs=1, type=int)
parser.add_argument('words', nargs='+')
args = parser.parse_args()

basedir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
strength_mat, id2word, word2id = generate_association_matrix.load_assoc_mat(
    basedir, args.input[0])

np.fill_diagonal(strength_mat, 1.)
n_selected = 0
indices = [word2id[w.upper()] for w in args.words]

for i in range(args.hops[0]):
    indices = np.where(np.sum(strength_mat[indices], axis=0))[0]

print 'Selected {} words.'.format(len(indices))

strength_mat = generate_association_matrix.tr_normalize(
    strength_mat[indices][:, indices])
id2word = np.asarray(id2word)[indices]
word2id = {w: i for i, w in enumerate(id2word)}

output_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
generate_association_matrix.save_assoc_mat(
    output_dir, args.output[0], strength_mat, id2word, word2id)
