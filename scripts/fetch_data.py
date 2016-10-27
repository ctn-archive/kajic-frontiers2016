#!/usr/bin/env python

import argparse

from sparat.datasets import datasets, get_dataset_path, fetch

parser = argparse.ArgumentParser(
    description="Fetch raw data files.",
    epilog="available datasets: " + ", ".join(datasets.keys()))
parser.add_argument(
    'tofetch', nargs='*',
    help="Names of datasets to fetch. If not given, all will be fetched.")
args = parser.parse_args()

if len(args.tofetch) <= 0:
    args.tofetch = datasets.keys()

for dataset_name in args.tofetch:
    fetch(dataset_name, get_dataset_path(dataset_name, datasets[dataset_name]))
