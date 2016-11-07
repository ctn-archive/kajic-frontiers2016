from __future__ import print_function

from model import SpaRat
import pandas as pd
import os
import time
import timeit

nr_prob = 25        # number of RAT problems
seed_start = 0      # number of seeds corresponds to the number of
seed_end = 55       # participants
sim_len = 10.       # simulation duration
assoc_th = 0.1      # cut-off association threshold
cue_strength = 0.1  # strenghts from individual cues
th = 0.6            # fraction of associations removed
d = 2048            # dimensionality of vectors

base_dir = os.path.dirname(__file__)
fname = time.strftime('%m%d_%H%M%S')
results_dir = os.path.join(base_dir, 'data', fname)

# load problems
problems_fname = '{}_problems_smith.csv'.format(nr_prob)
assoc_name = 'freeassoc_undir_symmetric'  # 'subset_{}'.format(nr_prob)

problems_dir = os.path.join(base_dir, os.pardir, os.pardir,
                            'data', 'rat', problems_fname)

prob = pd.read_csv(problems_dir, header=None, delimiter=' ',
                   index_col=0, names=[u'cue1', u'cue2', u'cue3', u'solution'])

try:
    import nengo_ocl
except ImportError:
    backend = 'nengo'
else:
    backend = 'nengo_ocl'

# run simulations
filenames = []
for prob_id, row in prob.iterrows():
    for seed in range(seed_start, seed_end+1):
        start = timeit.time.time()
        SpaRat().run(d=d, cue1=row.cue1, cue2=row.cue2, cue3=row.cue3,
                     solution=row.solution, vocab_seed=seed,
                     assoc_th=assoc_th,
                     th=th,
                     cue_strength=cue_strength,
                     sim_len=sim_len,
                     prob_id=prob_id,
                     assoc_name=assoc_name,
                     data_dir=results_dir,
                     backend=backend)
        end = timeit.time.time()
        print('Finished after {:.2f} sec.'.format(end-start))

# load simulation results in dataframe
cols_in = ['_prob_id', '_cue1', '_cue2', '_cue3', '_solution', 'correct',
           'responses']
cols_out = [c[1:] if c.startswith('_') else c for c in cols_in]

raw_data = cb.Data(path=results_dir)

sim_res = pd.DataFrame(raw_data.data, columns=cols_in)
sim_res.rename(columns=dict(zip(cols_in, cols_out)), inplace=True)


def replace_cues_with_ms(x, ms='_'):
    return [ms if r in (x.cue1, x.cue2, x.cue3) else r for r in x.responses]

sim_res['responses'] = sim_res.apply(lambda x: replace_cues_with_ms(x), axis=1)
sim_res['responses'] = sim_res['responses'].apply(lambda x: ",".join(x))

# save simulation resutls
datapath = os.path.join(base_dir, os.pardir, os.pardir,
                        'data', 'responses')

sim_res.to_csv(os.path.join(datapath, fname + '.csv'), index=False)
