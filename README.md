# A Spiking Neuron Model of Word Associations for the Remote Associates Test
This repository contains the source code of a model presented in:

Kajić, I., Gosmann, J., Stewart, T., Wennekers, T., Eliasmith, C.: 
"A Spiking Neuron Model of Word Associations for the Remote Associates Test"

It also contains step-wise instructions explaining how to download and install software needed to to reconstruct figures and run the model. 

## Instructions for reproducing the results and plots
Current installation instructions assume some level of familiarity with the
Python packaging system, experience with git and the command-line.

### Software Requirements
The project has been written in the Python and has been supports Python 2.7 and
3.5 or later (earlier versions of Python 3 are untested). It requires a number
of Python packages which should be automatically installed with the installation
of this project. On some systems (e.g., Windows) it might be necessary to
install NumPy and SciPy manually.

List of all requirements

* [Python](https://www.python.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [pandas](http://pandas.pydata.org/)
* [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/)
* [doit](http://pydoit.org/)
* [jupyter](http://jupyter.org/)
* [joblib](https://pythonhosted.org/joblib/)
* [sklearn](http://scikit-learn.org/stable/index.html)
* [Nengo](https://github.com/nengo/nengo) [2.2](https://github.com/nengo/nengo/releases/tag/v2.2.0)
* [pytry](https://github.com/tcstewar/pytry)


### Installation

2. Install NumPy and SciPy, e.g.
```bash
pip install numpy
pip install scipy
```
1. Clone and install this repository, e.g.
```bash
git clone https://github.com/ctn-archive/kajic-frontiers2016.git`
cd kajic-frontiers2016
pip install -e .
```
Note that it has this package has to be installed as “editable” (`-e`),
otherwise certain paths will not be found.

### Fetching and processing association data
Getting and processing the data can take a long time (up to few hours).

This project uses data available from other online sources:
- [Free Association Norms](http://w3.usf.edu/FreeAssociation/)
- [Google n-gram data](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html)

To fetch this data and generate corresponding matrices used in the paper, run
the script:
```
doit
```
in the cloned repository. This script will fetch the data from the
corresponding sources and generate matrices in the following folders:

Free association norms and google n-gram: `./data/associationmatrices/`

Vector representations of free norms and n-gram data: `./data/wordvectors/`

The raw Google n-grams are over 120 GB, for that reason the default setting in
the script will not attempt to download the raw data. Instead, it gets the
processed data stored on
[figshare](https://dx.doi.org/10.6084/m9.figshare.2066799) which is around 200
MB.

If this step has been successful, you should see the following output (the order
of some of the lines might differ):

```
.  fetch_data:freeassoc
.  fetch_data:google-processed
.  gen_association_matrices:freeassoc_symmetric
.  gen_association_matrices:freeassoc_asymmetric
.  gen_association_matrices:freeassoc_symmetric2
.  gen_association_matrices:freeassoc_undir_symmetric
.  gen_association_matrices:freeassoc_undir_asymmetric
.  gen_association_matrices:google_combined
.  gen_association_matrices:google_normalized
.  gen_association_matrices:google_symmetric
.  was:freeassoc_symmetric2_300
.  word_subsets:subset_1
.  word_subsets:subset_3
.  word_subsets:subset_5
```

### Reproducing figures and tables
Figures 3 and 4 are generated in the Notebook "RAT Response Filtering". Because
some parts of the figures require human data (such as distribution of
responses), which we do not have permission to publish, not all parts can be
reproduced.

To reproduce the numbers relating to model statistics in Table 2, run the
analysis script in the `analyses` directory:

```
python analyses/analyser.py fbinary_google_symmetric_simulations
```

`fbinary_google_symmetric_simulations` references the csv file produced by 
the `RAT Response Filtering` notebook.

### Running the model

Run the model with the default settings for the RAT problem "swiss, cottage,
cake" with:

```bash
python sparat/model/model.py
```

This simulation uses a complete dictionary with more than 5000 words, which can
take a long time to simulate. If you want to see a smaller example with
a subset of associations, you can set model parameters in model.py in the
following way:

```bash
python sparat/model/model.py --assoc_name subset_3 --d 512 --th 0 --cue1 demon --cue2
limit --cue3 way --solution speed
```

This sets the RAT problem with the cues `demon, limit, way` and the solution
`speed`, using the vocabulary `subset_3` which contains only associations of the
three cue words (and thus less words than the complete dictionary).  The
`--th`argument is used to set the threshold which determines the fraction of
associations to be removed and `--d` argument sets the dimensionality of
word vectors.

You can get a listing of all arguments/model parameters with
```bash
python sparat/model/model.py --help
```

After a successful run, the output should be similar to this one:

```
running SpaRat#20160822-213054-363903ad
Ones before flipping: 162.184232794
Ones after flipping 0.00: 162.184232794
Creating new pointers.

Done!
Simulation finished in 0:08:04.
_vocab_seed = 1
_d = 512
_assoc_dir = '../../data/associationmatrices'
_assoc_name = 'subset_3'
_th = 0
_cue1 = 'demon'
_cue2 = 'limit'
_cue3 = 'way'
_solution = 'speed'
_prob_id = 1
_cue_strength = 0.1
_sim_len = 10.0
_assoc_th = 0.05
_wta_feedback_strength = 0.5
_noise_std = 0.01
_use_saved_vocab = False
_noise_wta = 0.0
_integrator_feedback = 0.95
_backend = 'nengo'
_dt = 0.001
_seed = 1
_hide_overlay = False
_gui = False
correct = False
responses = ['STREET', 'DEVIL', 'BOUNDARY']
timings = [0.063, 0.27600000000000002, 2.504]
```

This output has also been stored in the `./data` under the name `SpaRat#...`
which is indicated in the simulation output. We created smaller sets of
problems which run faster than the whole set for the first three and first five
RAT problems (to see those, take look in `data/rat` repository in the main
project directory). To try another problem, change the cues, the solution
and the `assoc_name` parameter to either `subset_3` or `subset_5`, depending on
the index of the problem in the `25_problems_smith.csv` list (if problem index
is <=3 then use `subset_3`, else if problem id is between 3 and 5 use `subset_5`).

Model parameters used to run the model in the paper can be found in 
`run_models.py`. A set of simulations run with
`run_models.py` is also stored in `data` in a directory labeled with the
current date and time. For more information about model outputs, look at the
README in the data directory.

## Repository organization

### analyses
Qualitative semantic analyses of responses.

### data
Data files not included in the directory and are either downloaded from
external resources, generated by scripts in the repository or generated with
the model.

### notebooks
Jupyter notebooks with data analyses and plotting.

### scripts
Data processing scripts meant to be invoked from the command line.

### sparat
Python source code for processing the data and the model excluding command line
tools.
