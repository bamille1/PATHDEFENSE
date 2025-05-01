# PATHDEFENSE

This repository contains code accompanying the paper "Defense Against Shortest
Path Attacks" at the 2025 SIAM International Conference on Data Mining


## Dataset processing
To process the real datasets, download the following files:
* http://opsahl.co.uk/tnet/datasets/USairport500.txt (US Airports)
* https://datadryad.org/stash/downloads/file_stream/4588 (UK Metro, unzip the file that downloads)
* http://www.sociopatterns.org/files/datasets/003/ht09_contact_list.dat.gz (Hypertext 2009, unzip)
* http://snap.stanford.edu/data/as-733.tar.gz (Autonomous System, unzip)

Data preprocessing code is provided in the Jupyter notebook `dataset preprocessing.ipynb`, which can be run after extracting the datasets above.

## Experiments

Experiments in the paper were run using the following command:

      `python opt_defense.py <graphName> <trialInd> <method> <num_paths> <terminals> <success_factor> <budget_factor> <resultDir> <dataDir>`

where the argument values are set as follows:
* `<graphName>`:      a string denoting the dataset; can take the values 'er', 'ba', 'ws', 'sbm', 'as', 'airport', 'ht', and 'metro'
* `<trialInd>`:       index for distinguishing between trials (random number seed depends on this)
* `<method>`:         string denoting the optimization method; can take values 'zs' for the zero-sum method or 'heur' for PATHDEFENSE heuristic method
* `<num_paths>`:      the number of potential target paths
* `<terminals>`:      how to select the terminals of potential target paths---"same" for all target paths having the same terminal nodes, "different" if they are selected independently for each target path, and "community" to consider extra-community paths (with different terminal nodes for each); note that "community" can only be used with the 'sbm' and 'as' datasets
* `<success_factor>`: a number by which to multiply the defender cost when there is no attack and no perturbation to get the attack success cost \lambda
* `<budget_factor>`:  a number by which to multiply the average required budget of the attacker when no perturbation occurs to get the rate parameter for the budget distribution
* `<resultDir>`:      directory to save results
* `<dataDir>`:        directory holding the files airport.pkl, metro.pkl, ht09.pkl, and as19991211.pkl, which are created by the Jupyter notebook noted above

Experiments in the paper used trialInd of 0-9, both methods, num_paths set to powers of 2 from 2^0 to 2^3, all values for terminals (for the datasets where applicable), and (success_factor, budget_factor) pairs of (0.5, 1), (0.5, 2), and (0.1, 1).

In addition, we provide a small demonstration in the Jupyter notebook PATHDEFENSE demo.ipynb.

Note that all code requires a Gurobi licence to run on graphs used in the paper: https://www.gurobi.com/.

## Acknowledgements and Disclaimers

This material is based upon work supported by the United States Air Force
under Air Force Contract No. FA8702-15-D-0001 and the Combat Capabilities
Development Command Army Research Laboratory (under Cooperative Agreement
Number W911NF-13-2-0045).  Any opinions, findings, conclusions or
recommendations expressed in this material are those of the authors and do not
necessarily reflect the views of the United States Air Force or Army Research
Laboratory.

Copyright (C) 2023
Benjamin A. Miller, Zohair Shafi, Wheeler Ruml, Yevgeniy Vorobeychik, Tina
Eliassi-Rad, and Scott Alfeld

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
Part 252.227-7013 or 7014 (Feb 2014).  Notwithstanding any copyright notice,
U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS
252.227-7014 as detailed above.  Use of this work other than as specifically
authorized by the U.S. Government may violate any copyrights that exist in
this work
