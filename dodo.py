import os.path
import pandas as pd

from sparat.datasets import datasets, get_dataset_path


def assocmat_files(name):
    assocdir = os.path.join('data', 'associationmatrices')
    return [os.path.join(assocdir, name + ext) for ext in ['.npy', '.pkl']]


def sp_files(name):
    spdir = os.path.join('data', 'wordvectors')
    return [os.path.join(spdir, name + ext) for ext in ['.npy', '.pkl']]


def task_fetch_data():
    for name, dataset in datasets.items():
        if name == 'google':
            continue
        yield {
            'name': name,
            'actions': ['scripts/fetch_data.py ' + name],
            'targets': [
                os.path.join(get_dataset_path(name, dataset), file_['name'])
                for file_ in dataset['files']],
            'uptodate': [True],  # only download data if not existent
        }


def task_gen_association_matrices():
    matrices = [
        ('freeassoc', 'symmetric'),
        ('freeassoc', 'asymmetric'),
        ('freeassoc', 'symmetric2'),
        ('freeassoc', 'undir_symmetric'),
        ('freeassoc', 'undir_asymmetric')]
    for dataset, method in matrices:
        yield {
            'name': dataset + '_' + method,
            'actions': [
                'scripts/generate_association_matrix.py {0} {1}'.format(
                    dataset, method)],
            'file_dep': [
                os.path.join('data', 'raw', dataset, file_['name'])
                for file_ in datasets[dataset]['files']],
            'targets': assocmat_files(dataset + '_' + method),
        }

    yield {
        'name': 'google_combined',
        'actions': [
            'scripts/transform_association_matrix.py add google_combined '
            'google_bigrams google_1grams'],
        'file_dep': assocmat_files('google_bigrams') + assocmat_files(
            'google_1grams'),
        'targets': assocmat_files('google_combined'),
    }

    yield {
        'name': 'google_normalized',
        'actions': [
            'scripts/transform_association_matrix.py normalize '
            'google_normalized google_combined'],
        'file_dep': assocmat_files('google_combined'),
        'targets': assocmat_files('google_normalized'),
    }

    yield {
        'name': 'google_symmetric',
        'actions': [
            'scripts/transform_association_matrix.py symmetrify '
            'google_symmetric google_normalized'],
        'file_dep': assocmat_files('google_normalized'),
        'targets': assocmat_files('google_symmetric'),
    }


def task_was():
    for assocmat in ['freeassoc_symmetric2']:
        for d in [300]:
            yield {
                'name': assocmat + '_' + str(d),
                'actions': [
                    'scripts/generate_word_vectors.py {0} svd_factorize '
                    '{1}'.format(assocmat, d)],
                'file_dep': assocmat_files(assocmat),
                'targets': sp_files(
                    assocmat + '_svd_factorize_5018w_' + str(d) + 'd'),
            }

def task_word_subsets():
    mat = 'freeassoc_symmetric'
    rat_path = os.path.join(
        os.path.dirname(__file__), 'data', 'rat',
        '25_problems_smith.csv')
    rats = pd.read_csv(rat_path, header=None, delimiter=' ', index_col=0,
            names=[u'cue1', u'cue2', u'cue3', u'solution'])
    for nr_prob in [1, 3, 5]:
        cue_words = ' '.join(rats   [:nr_prob].values.flatten()).lower()
        name = 'subset_' + str(nr_prob)
        yield {
            'name': name,
            'actions': [
                'scripts/subset.py {0} subset_{1} 1' 
                ' {2}'.format(mat, nr_prob, cue_words)],
            'file_dep': assocmat_files(mat),
            'targets': assocmat_files(name)
        }

