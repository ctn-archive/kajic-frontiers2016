#!/usr/bin/env python

import os
import os.path
import sys
import argparse

from sparat.data_processing import generate_association_matrix


parser = argparse.ArgumentParser(
    description="Transform an association matrix.")
parser.add_argument(
    'method', nargs=1, help="Method to transform.")
parser.add_argument('output', nargs=1)
parser.add_argument('input', nargs='+')
args = parser.parse_args()

basedir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
strength_mat, id2word, word2id = zip(
    *(generate_association_matrix.load_assoc_mat(
        basedir, inp) for inp in args.input))
method = getattr(generate_association_matrix, 'tr_' + args.method[0])
transformed = method(*strength_mat)

assert all(list(x) == list(id2word[0]) for x in id2word)

output_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
generate_association_matrix.save_assoc_mat(
    output_dir, args.output[0], transformed, id2word[0], word2id[0])
