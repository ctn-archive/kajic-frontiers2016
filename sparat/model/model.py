from __future__ import print_function

import argparse
from collections import OrderedDict
import numpy as np
import os.path
import pickle
import re
import sys

from nengo import spa
import nengo
import pytry

from sparat import load_assoc_mat


def sanitize(w):
    return re.sub(r'\W', '_', w)


def ThresholdingPreset(threshold):
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].dimensions = 1
    config[nengo.Ensemble].intercepts = nengo.dists.Exponential(
        0.15, threshold, 1.)
    config[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
    config[nengo.Ensemble].eval_points = nengo.dists.Uniform(threshold, 1.)
    return config


def RandomSelector(n_options, noise_std, seed, net=None):
    if net is None:
        net = nengo.Network(label="random selector", seed=seed)

    with net:
        with nengo.Network(label="noise", seed=seed):
            net.noise = [
                nengo.Node(
                    nengo.processes.WhiteNoise(
                        dist=nengo.dists.Gaussian(mean=0., std=noise_std),
                        seed=seed),
                    label="WhiteNoise " + str(i + 1))
                for i in range(n_options)]
        net.selector = nengo.networks.AssociativeMemory(
            np.eye(n_options), threshold=.0, label="wta")
        net.selector.add_wta_network()
        nengo.Connection(net.selector.output, net.selector.input)
        for i, n in enumerate(net.noise):
            nengo.Connection(n, net.selector.input[i])

        net.invert = nengo.networks.EnsembleArray(
            50, n_options, label='invert')
        nengo.Connection(
            nengo.Node(1, label="bias"), net.invert.input,
            transform=[[1]] * n_options)
        for i in range(n_options):
            nengo.Connection(
                net.selector.output[i], net.invert.ensembles[i].neurons,
                transform=[[-1]] * net.invert.ensembles[i].n_neurons)

        net.reset = nengo.Node(size_in=1)
        for ens in net.selector.ensembles:
            nengo.Connection(
                net.reset, ens.neurons, transform=[[-1]] * ens.n_neurons,
                synapse=None)

        net.inv_output = net.invert.output

    return net


class CueSelection(spa.module.Module):
    def __init__(
            self, p, vocab, label=None, seed=None, add_to_container=None):
        super(CueSelection, self).__init__(
            label, seed, add_to_container)

        self.vocab = vocab

        with self:
            self.n_cues = 3
            self.cues = [spa.State(p.d) for i in range(self.n_cues)]
            for i, cue in enumerate(self.cues, 1):
                name = 'cue' + str(i)
                cue.label = name
                setattr(self, name, cue.input)

            self.selector = RandomSelector(self.n_cues, p.noise_std,
                                           p.vocab_seed)
            for i, cue in enumerate(self.cues):
                for ens in cue.state_ensembles.ensembles:
                    nengo.Connection(
                        self.selector.inv_output[i], ens.neurons,
                        transform=[[-1]] * ens.n_neurons)

            with nengo.Network(label="cue switching timer", seed=seed):
                with ThresholdingPreset(0.9):
                    self.switch = nengo.Ensemble(50, 1)
                    self.reset_ramp = nengo.Ensemble(50, 1)
                self.ramp = nengo.Ensemble(50, 1)
                self.bias = nengo.Node(1, label='bias')

                nengo.Connection(self.bias, self.ramp, transform=0.1)
                nengo.Connection(self.ramp, self.ramp, synapse=0.1)
                nengo.Connection(self.ramp, self.switch)
                nengo.Connection(self.reset_ramp, self.ramp, transform=-5)

            nengo.Connection(self.switch, self.selector.reset)
            nengo.Connection(
                self.selector.inv_output, self.reset_ramp,
                transform=[[0.33] * 3], synapse=0.1)

            self.output = nengo.Node(size_in=p.d, label="output")
            for cue in self.cues:
                nengo.Connection(cue.output, self.output)

        self.inputs = {'cue' + str(i): (cue.input, self.vocab)
                       for i, cue in enumerate(self.cues, start=1)}
        self.outputs = {'default': (self.output, self.vocab)}


class SpaRat(pytry.NengoTrial):
    def params(self):
        self.param("vocab seed", vocab_seed=1)
        self.param("vocab dimensionality", d=2048)
        self.param("association matrix dir", assoc_dir=os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'data',
            'associationmatrices'))
        self.param("association matrix name",
                     assoc_name='freeassoc_undir_symmetric')
        self.param("percentage of removed associations", th=0.3)
        self.param("cue word", cue1='cake')
        self.param("cue word", cue2='swiss')
        self.param("cue word", cue3='cottage')
        self.param("solution", solution='cheese')
        self.param("problem id", prob_id=1)
        self.param("connection strength between cues and response",
                     cue_strength=0.1)
        self.param("simulation length", sim_len=10.)
        self.param("associative memory threshold", assoc_th=0.05)
        self.param("wta feedback strength", wta_feedback_strength=0.5)
        self.param("cue selector noise std", noise_std=0.01)
        self.param("use saved vocab", use_saved_vocab=False)
        self.param("wta associative memory noise input std", noise_wta=0.0)
        self.param("inhibition integrator feedback", integrator_feedback=.95)

    def model(self, p):
        assoc, id2word, _ = load_assoc_mat(p.assoc_dir, p.assoc_name)

        cues = set([c.upper() for c in [p.cue1, p.cue2, p.cue3]])
        assert all(c in id2word for c in cues)

        np.fill_diagonal(assoc, 0.)

        rng = np.random.RandomState(p.vocab_seed)

        print('Ones before flipping:', assoc.sum())

        # find all ones
        one_idx = np.where(assoc > 0)

        # how many of these ones should be flipped
        nr_ones = len(one_idx[0])
        nr_flip = int(p.th*nr_ones)
        to_flip_idx = rng.randint(0, nr_ones, size=nr_flip)

        # flip those to 0 (remove associations)
        assoc[(one_idx[0][to_flip_idx], one_idx[1][to_flip_idx])] = 0

        print('Ones after flipping {0:.2f}: {1}'.format(p.th, assoc.sum()))

        if not p.use_saved_vocab:
            print('Creating new pointers.')
            self.vocab = spa.Vocabulary(
                p.d, rng=np.random.RandomState(p.vocab_seed))
            for w in id2word:
                self.vocab.parse(sanitize(w))
        else:
            sp_dir = os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, 'data',
                'spa_vocabularies')
            fname = 'vocab_{0}d_{1}s.pkl'.format(p.d, p.vocab_seed)
            print('Loading existing pointers:', fname)
            voc_path = os.path.join(sp_dir, fname)
            with open(voc_path, 'rb') as f:
                self.vocab = pickle.load(f)
        print('Done!')

        with spa.SPA(seed=p.seed, vocabs=[self.vocab]) as model:
            model.state = spa.State(p.d, vocab=self.vocab.create_subset(cues))
            model.integrator = spa.State(p.d, feedback=p.integrator_feedback)
            model.wta = spa.AssociativeMemory(
                self.vocab, threshold=p.assoc_th, threshold_output=True,
                wta_output=True)

            with ThresholdingPreset(0):
                model.inh_array = nengo.networks.EnsembleArray(
                    n_neurons=50, n_ensembles=len(self.vocab.vectors))

            # model input
            model.cue_selection = CueSelection(
                p, seed=p.seed, vocab=self.vocab)
            nengo.Connection(model.cue_selection.output, model.state.input)

            model.stimulus = spa.Input(
                cue_selection_cue1=p.cue1.upper(),
                cue_selection_cue2=p.cue2.upper(),
                cue_selection_cue3=p.cue3.upper())

            assoc_tr = np.dot(
                self.vocab.vectors.T, np.dot(assoc.T, self.vocab.vectors))

            # cue associates at the input of the wta
            nengo.Connection(
                model.state.output, model.wta.input, transform=0.7*assoc_tr)

            # influence from other cues
            nengo.Connection(
                model.cue_selection.cue1, model.wta.input,
                transform=p.cue_strength*assoc_tr)

            nengo.Connection(
                model.cue_selection.cue2, model.wta.input,
                transform=p.cue_strength*assoc_tr)

            nengo.Connection(
                model.cue_selection.cue3, model.wta.input,
                transform=p.cue_strength*assoc_tr)

            # stabilizing feedback connection
            nengo.Connection(model.wta.output, model.wta.input)

            # current response associates as the input
            nengo.Connection(
                model.wta.output, model.wta.input,
                transform=p.wta_feedback_strength*assoc_tr, synapse=0.1)

            # white noise input to the WTA
            if p.noise_wta > 0:
                print('Adding noise to WTA.')
                wn = nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=0., std=p.noise_wta))
                model.wta_noise = nengo.Node(wn, size_out=p.d)

                nengo.Connection(model.wta_noise, model.wta.input)

            # response inhibition
            nengo.Connection(
                model.wta.output, model.integrator.input, synapse=0.1)

            nengo.Connection(
                model.integrator.output, model.inh_array.input,
                transform=self.vocab.vectors)

            for pre, post in zip(
                    model.inh_array.ensembles, model.wta.am.am_ensembles):
                nengo.Connection(pre, post, transform=-2)

            self.probe = nengo.Probe(model.wta.output, synapse=0.03)
            self.probe_inp = nengo.Probe(model.wta.input, synapse=0.03)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.sim_len)

        min_sim = 0.8   # discard responses while model initializes
        similarities = spa.similarity(sim.data[self.probe], self.vocab)

        mid = np.argmax(similarities, axis=1)
        sim_max = np.max(similarities, axis=1)

        responses = mid[sim_max > min_sim]
        timing = sim.trange()[sim_max > min_sim]

        unique_responses = OrderedDict()
        for t, r in zip(timing, responses):
            word = self.vocab.keys[r]
            if word not in unique_responses:
                unique_responses[word] = t

        correct = p.solution.upper() in unique_responses.keys()
        return {
            'responses': unique_responses.keys(),
            'timings': unique_responses.values(),
            'correct': correct}


def str2bool(text):
    return text.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    trial = SpaRat()
    params = trial.parse_args(sys.argv[1:])
    trial.run(**params)
