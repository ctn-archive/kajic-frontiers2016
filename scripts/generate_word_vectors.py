#!/usr/bin/env python

import os
import os.path
import sys
import argparse

from sparat import dimension_reduction
from sparat import load_assoc_mat
from sparat import spgen

parser = argparse.ArgumentParser(description="Generate word vectors.")
parser.add_argument(
    'association_matrix', nargs=1,
    help="Association matrix to generate vectors from.")
parser.add_argument(
    'dimred', nargs=1, help="Dimension reduction method.")
parser.add_argument(
    'd', nargs=1, type=int, help="Dimensionality of word vectors.")
args = parser.parse_args()

input_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
strength_mat, id2word, word2id = load_assoc_mat(
    input_dir, args.association_matrix[0])

pointers = spgen.from_assoc_matrix(
    strength_mat, getattr(dimension_reduction, args.dimred[0]), args.d[0])

output_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'wordvectors')
name = '{assoc}_{dimred}_{words}w_{d}d'.format(
    assoc=args.association_matrix[0], dimred=args.dimred[0],
    words=pointers.shape[0], d=args.d[0])
spgen.save_pointers(output_dir, name, pointers, id2word, word2id)
