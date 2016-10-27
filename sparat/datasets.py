"""
Downloads the Norms and Ngram datasets.
"""
import sys

import itertools
import os
import os.path
if sys.version_info[0] >= 3:
    import urllib.request as urllib
else:
    import urllib
import string
import time
import webbrowser

import pandas as pd


def get_dataset_path(name, dataset):
    if 'path' in dataset:
        path = dataset['path']
    else:
        path = os.path.join(os.path.dirname(
            __file__), os.pardir, 'data', 'raw', name)
    return path


def load_free_association_data(path, column='FSG'):
    """Loads the free association data from `path`.

    Parameters
    ----------
    path : str
        Path to load data from.
    column : str, optional
        Column in the data files that gives the association strength. Use 'FSG'
        for forward strength and 'BSG' for backward strength.

    Returns
    -------
    A tuple (words, association_database) where words is a set of all occuring
    words (either as cue or target) and association_database is a list of
    associations. Each association is a tuple (cue, target, strength).
    """
    normed_responses = 0
    words = set()
    association_database = []

    filenames = [f['name'] for f in datasets['freeassoc']['files']]
    for filename in filenames:
        df = pd.read_csv(
            os.path.join(path, filename), skipinitialspace=True,
            comment='<',  # comment='<' is a hackish way to skip HTML tags
            encoding='windows-1252')
        df[column] = pd.to_numeric(df[column])

        df_normed = df[df['NORMED?'] == 'YES']
        normed_responses += len(df_normed)

        # extract norms
        for _, row in df_normed.iterrows():
            cue, target = row['CUE'].upper(), row['TARGET'].upper()
            words.add(cue)
            words.add(target)

            strength = row[column]
            association_database.append((cue, target, strength))

    assert len(words) == 5018, "Number words should be 5018."
    assert normed_responses == 63619, \
        "Number of normed responses should be 63619"

    return words, association_database


datasets = {
    'google-processed': {
        'description': 'Processed Google n-gram dataset',
        'moreinfo': 'See CogSci paper.',
        'path': os.path.join(
            os.path.dirname(__file__), os.pardir, 'data',
            'associationmatrices'),
        'files': [
            {
                'name': 'google_1grams.npy',
                'url': 'https://ndownloader.figshare.com/files/3676104'
            },
            {
                'name': 'google_1grams.pkl',
                'url': 'https://ndownloader.figshare.com/files/3676107'
            },
            {
                'name': 'google_bigrams.npy',
                'url': 'https://ndownloader.figshare.com/files/3676110'
            },
            {
                'name': 'google_bigrams.pkl',
                'url': 'https://ndownloader.figshare.com/files/3676113'
            },
        ]
    },
    'google': {
        'description': 'Google n-gram dataset',
        'moreinfo':
            'http://storage.googleapis.com/books/ngrams/books/datasetsv2.html',
        'files': [
            {
                'name': 'googlebooks-eng-all-1gram-20120701-' + c + '.gz',
                'url':
                    'http://storage.googleapis.com/books/ngrams/books/'
                    'googlebooks-eng-all-1gram-20120701-' + c + '.gz',
            }
            for c in string.ascii_lowercase
        ] + [
            {
                'name': 'googlebooks-eng-all-2gram-20120701-' + c1 + c2 + '.gz',
                'url':
                    'http://storage.googleapis.com/books/ngrams/books/'
                    'googlebooks-eng-all-2gram-20120701-' + c1 + c2 + '.gz',
            }
            for c1, c2 in itertools.product(
                string.ascii_lowercase, string.ascii_lowercase)
        ]
    },
    'freeassoc': {
        'description': 'University of South Florida Free Association Norms',
        'moreinfo': 'http://w3.usf.edu/FreeAssociation/',
        'loader': load_free_association_data,
        'files': [
            {
                'name': 'Cue_Target_Pairs.A-B',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.A-B',
            }, {
                'name': 'Cue_Target_Pairs.C',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.C',
            }, {
                'name': 'Cue_Target_Pairs.D-F',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.D-F',
            }, {
                'name': 'Cue_Target_Pairs.G-K',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.G-K',
            }, {
                'name': 'Cue_Target_Pairs.L-O',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.L-O',
            }, {
                'name': 'Cue_Target_Pairs.P-R',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.P-R',
            }, {
                'name': 'Cue_Target_Pairs.S',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.S',
            }, {
                'name': 'Cue_Target_Pairs.T-Z',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.T-Z',
            }
        ],
    },
}


def report_progress(blocks_transferred, block_size, total_size):
    sys.stdout.write("\r{0}/{1} KiB".format(
        blocks_transferred * block_size // 1024, total_size // 1024))
    sys.stdout.flush()


def fetch(dataset_name, path=os.curdir):
    if not os.path.exists(path):
        os.makedirs(path)

    for file_ in datasets[dataset_name]['files']:
        sys.stdout.write("Fetching " + file_['name'] + os.linesep)
        if file_.get('manual', False):
            sys.stderr.write("Downloading this file needs user intervention.")
            sys.stderr.write(os.linesep)
            sys.stderr.write("A browser with the download page will open.")
            sys.stderr.write(os.linesep)
            sys.stderr.write("Please save the file to:")
            sys.stderr.write(os.linesep)
            filename = os.path.abspath(os.path.normpath(
                os.path.join(path, file_['name'])))
            sys.stderr.write(filename)
            sys.stderr.write(os.linesep)
            webbrowser.open(file_['url'])
            while (not os.path.exists(filename) or
                   os.path.getsize(filename) < file_['size']):
                time.sleep(1)
        else:
            urllib.urlretrieve(
                file_['url'], os.path.join(path, file_['name']),
                report_progress)
        sys.stdout.write(os.linesep)
