#!/usr/bin/env python

import os
import os.path
import argparse

from sparat.datasets import datasets, load_free_association_data
from sparat.data_processing import generate_association_matrix


parser = argparse.ArgumentParser(
    description="Read association data and write an association matrix.")
parser.add_argument(
    '-s', '--stats', action='store_true',
    help="Show statistics about the association matrix.")
parser.add_argument(
    'dataset', nargs=1, help='freeassoc or google')
parser.add_argument(
    'method', nargs=1,
    help="Method to generate the association matrix (undir=undirected):" +
    "symmetric, asymmetric, symmetric2, undir_symmetric, undir_asymmetric")
args = parser.parse_args()

basedir = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'raw')
input_dir = os.path.join(basedir, args.dataset[0])
if args.dataset[0] == 'google':
    data = load_free_association_data(os.path.join(basedir, 'freeassoc'))[:1]
else:
    data = datasets[args.dataset[0]]['loader'](input_dir)
method = getattr(generate_association_matrix, 'gen_' + args.method[0])
assoc_mat = method(*data)

if args.stats:
    generate_association_matrix.print_stats(assoc_mat[0])

output_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
name = "{dataset}_{method}".format(
    dataset=args.dataset[0], method=args.method[0])
generate_association_matrix.save_assoc_mat(output_dir, name, *assoc_mat)
