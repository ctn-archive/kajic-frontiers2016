This directory contains simulation data produced by the model. The simulations
are run with the script `run_models.py`, contained in the parent directory.
Every time a script is run, it creates a directory with the label of a format
`mmdd_hhmmss`(`mm` is the current month, `dd` current day, `hh` hour etc.). The
directory contains a list of text files of a format `SpaRat#2016...` and each
text file is a simulation of one RAT problem. Every text file contains
information about model parameters that were used in the simulation, problem
cues, the solution, 

This folder (i.e. simulation data) is also used in the notebook "RAT response
filtering" which filters the data and creates the output (in the
`../../data/responses` directory) for the analyser script.

Currently, the directory contains the simulation data that was used in the
paper. Each `th_0x` directory refers to a set of simulations where `x`
indicates percentage of associations removed in the FAN data. For example, in
'th_06' 60% of the associations were removed. `simulations` directory contains
the data from all three directories.

