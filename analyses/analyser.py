#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

from sparat import load_assoc_mat
from analyses_description import descriptions
from scipy.stats import binom_test
from sklearn.feature_selection import f_regression


class Analyser(object):
    """ Analyser class containing RAT response analysis methods.

    Methods have been adapted from: K.A. Smith et al., Cognition
    128 (2013), 64-75. The class expect

    Parameters
    ----------
    space : str
        Semantic space used for word similarity measures.
    data : str
        File name of response data in `data` directory. If it contains `human`
        in the filename, the source folder will be `human`, otherwise
        `simulations`.

    Attributes
    ----------
    W : np.array
        N x d matrix containing N words represented by d dimensions.
    i2w : list
        list of mappings between word ids and words.
    w2i : dict
        Dictionary of mappings between words and word ids.
    data_words : pandas DataFrame
        Loaded csv responses from the `data` file.
    data_idx : pandas DataFrame
        Same as data_words, but instead of words uses word indices, also
        includes a column with list of primary cues for every row.
    data_clean : pandas DataFrame
        Same as data_idx with duplicate responses and missing words removed.
    resutls : pandas DataFrame
        Analysis results.
    """

    def __init__(self, space, dataset, path):
        self.semantic_space = space
        self.dataset = os.path.join(path, dataset)

        self.W, self.i2w, self.w2i = self._load_semantic_space()

        # store response data as words and as word indices
        self.data_words = self._load_data()

        # missing data
        self.miss = -1
        self.data_idx = self._convert_words_to_idx()

        # assign a primary cue to every response
        self.data_idx = self._assign_primary_cues()

        # store cleaned version of the data
        self.data_clean = self._clean_data()

        self.results = self._initialize_results()

        self._compute_stats()

    def _assign_primary_cues(self):
        """
        For every word in response assign a primary cue as the cue with the
        greatest similarity. For missing reponses assing -1.
        """
        def assign_pc(x):
            pcs = []

            for rsp_idx in x['responses']:
                cues_idx = [x.cue1, x.cue2, x.cue3]

                cues_similarity = [self._get_similarity(rsp_idx, cue)
                                   for cue in cues_idx]
                primary_cue_idx = np.argsort(cues_similarity)[-1]

                # append -1 cue for missing data
                primary_cue = self.miss if rsp_idx < 0 else \
                    cues_idx[primary_cue_idx]

                pcs.append(primary_cue)

            x['primary cues'] = np.array(pcs, dtype=np.int)
            return x

        return self.data_idx.apply(func=assign_pc, axis='columns')

    def _clean_data(self):
        """
        Creates a new data set by dropping all missing entries.
        """
        cleaned = self.data_idx.copy(deep=True)

        for index, row in cleaned.iterrows():
            cleaned.set_value(index, 'responses',
                              filter(lambda x: x != self.miss, row.responses))
            cleaned.set_value(index, 'primary cues',
                              filter(lambda x: x != self.miss,
                                     row['primary cues']))
        return cleaned

    def _compute_stats(self):
        self.was_stats()
        self.avg_response_similarity()
        self.avg_response_similarity(self.data_clean)
        self.permute_pcs_similarity()
        self.across_cue_similarity()
        self.within_cue_similarity()
        self.rate_similarity_answer(True)
        self.baseline_vs_actual_cue_assignment()

    def _convert_words_to_idx(self):
        """
        Returns the data structure  of the same format as the one it's being
        called from. Instead of words it contains word indices from WAS.
        """
        def row_to_id(x):
            # convert problem words to word indices
            cols = ['cue1', 'cue2', 'cue3', 'solution']
            x[cols] = x[cols].apply(func=lambda y: self.w2i[y])

            # get a list of response words
            responses = x['responses'].strip().split(',')

            # get their indices or -1 if not in WAS
            response_idx = map(lambda y: self.w2i.get(y, self.miss), responses)
            x['responses'] = response_idx

            assert len(responses) == len(response_idx)

            return x

        data = self.data_words.copy(deep=True)
        return data.apply(func=row_to_id, axis='columns')

    def _fill_result_section(self, row, g1, g2):
        if row in ('same_cue_percentage'):
            same, pairs = g2[0], g2[1]
            g2 = 100*same/pairs

        m1, m2 = np.mean(g1), np.mean(g2)

        self.results.loc[row]['G1_data'] = np.array(g1)
        self.results.loc[row]['G1_avg'] = m1

        self.results.loc[row]['G2_data'] = np.array(g2)
        self.results.loc[row]['G2_avg'] = m2

        if row not in ('rate_with_distance', 'same_cue_percentage'):
            self.results.loc[row]['ttest'] = np.array(st.ttest_ind(g1, g2))

            s_mdiff = np.sqrt(np.var(g1)/len(g1) + np.var(g2)/len(g2))
            confidence = 0.95
            dof = len(g1) + len(g2) - 2
            t = st.t.ppf((1+confidence)/2., dof)

            # lower and upper bounds
            ll = m1 - m2 - t*s_mdiff
            ul = m1 - m2 + t*s_mdiff
            self.results.loc[row]['CI'] = [ll, ul]
            self.results.loc[row]['df'] = dof
        elif row in ('same_cue_percentage'):
            self.results.loc[row]['ttest'] = np.array(
                binom_test(same, pairs, p=g1/100.))

    def _get_similarity(self, w1, w2):
        # association matrix
        if self.W.shape[0] == self.W.shape[1]:
            s = self.W[w1, w2]
        else:  # cosine angle for vectors
            v1, v2 = self.W[w1], self.W[w2]
            s = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            if w1 < 0 or w2 < 0:
                s = 0
        return s

    def _initialize_results(self):
        cols = ['Analysis', 'Description', 'Group1', 'Group2', 'G1_data',
                'G2_data', 'G1_avg', 'G2_avg', 'ttest', 'CI', 'df']
        df = pd.DataFrame(columns=cols)

        for analysis in descriptions:
            df = df.append(analysis, ignore_index=True)

        return df.set_index('Analysis')

    def _invalid_pair(self, w1, w2, sol):
        """
        Checks whether one of the pair words is missing data, is a solution
        word or whether the words are equal.
        """
        return (w1 < 0 or w2 < 0) or (sol in (w1, w2)) or w1 == w2

    def _load_semantic_space(self):
        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, 'data')
        try:
            mat = load_assoc_mat(os.path.join(data_dir, 'associationmatrices'),
                                 self.semantic_space)
        except IOError:
            mat = load_assoc_mat(os.path.join(data_dir, 'wordvectors'),
                                 self.semantic_space)
        return mat

    def _load_data(self):
        return pd.read_csv(self.dataset, delimiter=',', header=0)

    def avg_response_similarity(self, data=None):
        """ Computes average similarity between neighbouring word pairs. If
        they have the same primary cue, the similarity is added to within cue
        cluster, otherwise to across cue cluster.
        """
        def compute_similarity(responses, pcs, solution):
            within_cue, across_cue = [], []

            for i in range(len(responses)-1):
                word1, word2 = responses[i], responses[i+1]

                if not self._invalid_pair(word1, word2, solution):
                    similarity = self._get_similarity(word1, word2)

                    if pcs[i] == pcs[i+1]:
                        within_cue.append(similarity)
                    else:
                        across_cue.append(similarity)

            return np.array(within_cue), np.array(across_cue)

        if data is None:
            data = self.data_idx
            analysis = 'avg_response_similarity'
        else:
            analysis = 'avg_response_similarity_clean'

        sims = data.apply(
            lambda x: compute_similarity(
                x['responses'], x['primary cues'], x['solution']),
            axis='columns')

        wc_all = [y for x in sims.apply(lambda x: x[0]) for y in x]
        ac_all = [y for x in sims.apply(lambda x: x[1]) for y in x]

        self._fill_result_section(analysis, wc_all, ac_all)

    def permute_pcs_similarity(self):
        """
        Compares adjacent response similarities for original cue assignment and
        shuffled cue assignment.

        - Type A responses: word pairs with the same primary cue in the
            original data set, and different primary cues in shuffled data set

        - Tybe B responses: word pairs with different primary cues in the
            original data set, and the same in shuffled data set

        """
        def assign_random_pc(row):
            """
            Returns the row-index of a different problem than `row` with
            the same number of responses. This is regarded as 'shuffled'
            response'.

            Rarely this crashes as no response with the same length can be
            found, this can be avoided by re-running the script.
            """
            resp_len = len(row.responses)

            # get lengths of all responses as a list
            resp_len_all = self.data_idx.responses.apply(
                lambda x: len(x))

            # drop same problem items
            to_drop = self.data_idx[self.data_idx.prob_id == row.prob_id].index
            resp_len_all.drop(to_drop, inplace=True)

            ok_indices = resp_len_all[resp_len_all == resp_len]

            index = -1  # return this if no other problem matches

            if len(ok_indices) > 0:
                index = np.random.choice(ok_indices.index)

            return index

        typeA, typeB = [], []
        for index, row in self.data_idx.iterrows():
            true_pc = row['primary cues']
            if len(true_pc) == 1 and -1 in true_pc:
                continue
            rnd_pc_row_idx = assign_random_pc(row)

            if rnd_pc_row_idx < 0:
                continue

            # random primary cues from another problem
            rnd_pc = self.data_idx['primary cues'][rnd_pc_row_idx]

            assert len(true_pc) == len(rnd_pc)    # equal length
            assert (true_pc - rnd_pc).sum() != 0  # not the same

            for i in range(len(row.responses)-1):
                word1, word2 = row.responses[i], row.responses[i+1]

                if not self._invalid_pair(word1, word2, row.solution):
                    similarity = self._get_similarity(word1, word2)
                    if true_pc[i] == true_pc[i+1] and rnd_pc[i] != rnd_pc[i+1]:
                        typeA.append(similarity)
                    elif true_pc[i] != true_pc[i+1] and\
                            rnd_pc[i] == rnd_pc[i+1]:
                        typeB.append(similarity)

        self._fill_result_section('permute_pcs_similarity', typeA, typeB)

    def across_cue_similarity(self):
        """
        Test for breaks between clusters by comparing adjacent and
        non-adjacent responses with different primary cues.

        Adjacent: neighboring responses with diff primary cues
        Non-adjacent: all non-adjacent responses with different primary cues
            before a cluster with the same primary cue
        """
        adjacent, non_adjacent = [], []

        for _, row in self.data_idx.iterrows():
            seq_len = len(row.responses)

            for i in range(seq_len-1):
                word1 = row.responses[i]
                for j in range(i+1, seq_len):
                    word2 = row.responses[j]

                    if not self._invalid_pair(word1, word2, row.solution):
                        if row['primary cues'][i] == row['primary cues'][j]:
                            break

                        similarity = self._get_similarity(word1, word2)
                        if i+1 == j:
                            adjacent.append(similarity)
                        else:
                            non_adjacent.append(similarity)

        self._fill_result_section('across_cue_similarity', adjacent,
                                  non_adjacent)

    def within_cue_similarity(self):
        """
        Test for sequential dependence of responses with same primary cue by
        comparing similarities between adjacent and non-adjacent responses.
        """
        adjacent, non_adjacent = [], []

        for _, row in self.data_idx.iterrows():
            seq_len = len(row.responses)

            for i in range(seq_len-1):
                word1 = row.responses[i]
                for j in range(i+1, seq_len):
                    word2 = row.responses[j]

                    if not self._invalid_pair(word1, word2, row.solution):
                        if row['primary cues'][i] != row['primary cues'][j]:
                            break

                        similarity = self._get_similarity(word1, word2)
                        if i+1 == j:
                            adjacent.append(similarity)
                        else:
                            non_adjacent.append(similarity)

        self._fill_result_section('within_cue_similarity', adjacent,
                                  non_adjacent)

    def rate_similarity_answer(self, doplots=False):
        n = 10
        means = np.zeros((2, n))
        errors = np.zeros((2, n))
        ok_idx = np.empty((2, n), dtype=np.bool)
        colors = ['denim blue', "pale red"]
        x_reg = []

        for i, case in enumerate(('True', 'False')):
            trials = self.data_idx.query('correct==%s' % case)

            sim_list = [[] for _ in range(n)]

            for j, responses in enumerate(trials.responses):
                ans = trials.iloc[j].solution if case == 'True' else\
                    responses[-1]

                pos_ans = responses.index(ans)
                pos_begin = max(pos_ans-n, 0)

                resp_chunk = np.array(responses[pos_begin:pos_ans])

                if len(resp_chunk) < 0:  # correct answer is the only one
                    continue

                row_sims = np.array(map(lambda x: self._get_similarity(x, ans),
                                        resp_chunk), dtype=np.float)

                # flip so that 0.element is the first before the solution
                row_sims = row_sims[::-1]
                resp_chunk = resp_chunk[::-1]
                assert len(resp_chunk) == len(row_sims)

                for k in range(len(row_sims)):
                    if resp_chunk[k] >= 0:  # skip missing data
                        sim_list[k].append(row_sims[k])

            means[i] = [np.mean(l) for l in sim_list]
            errors[i] = [np.std(l)/np.sqrt(len(l)) for l in sim_list]
            ok_idx[i] = ~np.isnan(means[i])

            n_samples = np.sum([len(l) for l in sim_list])
            print case, n_samples
            x = np.zeros((2, n_samples))

            l, xi = 0, -1
            for sublist in sim_list:
                slen = len(sublist)
                x[0, l: l+slen] = xi
                x[1, l: l+slen] = sublist
                l += slen
                xi -= 1
            # ipdb.set_trace()
            print 'Regression for >0 slope (n={0}): {1} p={2}'.format(
                n_samples, *f_regression(x[0].reshape(-1, 1), x[1]))
            x_reg.append(x)

        p = st.ttest_ind(x_reg[0][1], x_reg[1][1])[1]

        print 'T-test for similarity to final answer {0:.2f} vs {1:.2f}'\
            ' (p={2:.2f})'.format(
                np.mean(x_reg[0][1]), np.mean(x_reg[1][1]), p)

        n1, n2 = x_reg[0].shape[1], x_reg[1].shape[1]

        x = np.zeros((3, n1 + n2))
        y = np.zeros((n1 + n2))

        x[0, :n1] = x_reg[0][0]
        y[:n1] = x_reg[0][1]

        x[[0, 2], n1:] = x_reg[1][0]
        x[1, n1:] = 1
        y[n1:] = x_reg[1][1]

        print 'Regression for equal slopes (Fs, p-values):\n'
        print f_regression(x.T, y)

        if doplots:
            plt.figure()

        rates = np.zeros(2)
        for i, case in enumerate(('True', 'False')):
            means_i, errors_i = means[i, ok_idx[i]], errors[i, ok_idx[i]]

            if doplots:
                plt.errorbar(range(1, len(means_i)+1), means_i,
                             yerr=errors_i, label=case, marker='o',
                             color=sns.xkcd_rgb[colors[i]])
            x, y = np.arange(1, len(means_i)+1), means_i
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), color=sns.xkcd_rgb[colors[i]], alpha=0.5, lw=2)
            plt.xlabel('Response position prior to the final response')
            plt.ylabel('Similarity to final response')
            plt.xlim([0.9, n+.4])

            rates[i] = p[1]

        self._fill_result_section('rate_with_distance', rates[0],
                                  rates[1])
        if doplots:
            plt.legend(['Correct', 'Incorrect'])
            plt.show()

        return means

    def baseline_vs_actual_cue_assignment(self):
        """
        Computes the baseline and actual percentage of primary cue assignment.
        """
        same, nr_pairs = 0, 0

        cue_counts = np.zeros((len(self.data_clean), 3))
        for i, row in self.data_clean.iterrows():
            if len(row.responses) < 1:
                continue
            cue_mapping = {row['cue1']: 0, row['cue2']: 1, row['cue3']: 2}
            primary_cues = [
                cue_mapping.get(x, -1) for x in row['primary cues']]
            assert -1 not in primary_cues

            # for calculating baseline probability
            values, counts = np.unique(primary_cues, return_counts=True)
            cue_counts[i, values] = counts

            # for calculating actual probability
            same += np.sum(np.diff(primary_cues) == 0)
            nr_pairs += len(primary_cues) - 1

        baseline = np.sum(
            (np.sum(cue_counts, axis=0) / np.sum(cue_counts)) ** 2)

        self._fill_result_section('same_cue_percentage', 100*baseline,
                                  [float(same), nr_pairs])

    def was_stats(self):
        f_sim = self._get_similarity

        sim_res = []  # similarity to response
        sim_all = []  # similarity to random words

        total = 0
        for index, row in self.data_idx.iterrows():
            s, resp = row.solution, row.responses
            resp = filter(lambda x: x != s and x != -1, resp)

            if len(resp) > 0:
                all_words = set(np.arange(len(self.i2w)))-set([s])
                random = np.random.choice(list(all_words), len(resp))

                sim_all.append(np.mean(map(lambda w: f_sim(s, w), random)))
                sim_res.append(np.mean(map(lambda r: f_sim(s, r), resp)))

                total += len(resp)+len(random)

        print 'Similarity answer to responses: {:.3f}'.format(np.mean(sim_res))
        print 'Similarity answer to rnd words: {:.3f}'.format(np.mean(sim_all))
        print 't-test for difference (n={}): {:.2f} p={:.2f}'.format(
            total, *st.ttest_ind(sim_res, sim_all))

if __name__ == "__main__":
    pd.set_option('display.precision', 3)

    parser = argparse.ArgumentParser(
        description="Load RAT response data and print analyses.")
    parser.add_argument(
        'dataset', nargs=1, help='name of the dataset to be analyzed')
    parser.add_argument(
        '--space', nargs=1,
        help="Semantic space to be used for the similarity comparisons",
        default="freeassoc_symmetric2_svd_factorize_5018w_300d")
    parser.add_argument(
        '--path', nargs=1,
        help="Relative path to the location of the dataset (default: " +
        "../data/responses/)",
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, 'data',
            'responses')
    )

    args = parser.parse_args()

    analyser = Analyser(
        space=args.space,
        dataset=args.dataset[0] + '.csv',
        path=args.path)

    print analyser.results
