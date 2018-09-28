# The Self-Organizing Recurrent Neural Network (SORN)

A SORN repository for general purposes, containing a few experiments and examples. This repository is based on the original [SORN repository](https://github.com/chrhartm/SORN) by Christoph Hartmann combined with adaptations to new experiments.  

## Getting started with the repository

The scripts for each experiment are stored in different folders with the experiments' respective name. Each folder contain a minimum of three scripts: the experiment parameters (`param.py`), instructions (`experiment.py`), and the input details (`source.py`). The parameters in these scripts can be modified at will to reproduce various results. Additionally, each folder contains the relevant plot scripts, for visualizing the results.

To simulate a single experiment, run `python common/run_single.py <ExperimentName>` from the main folder.

Currently implemented experiments: 

* CountingTask (from Lazar et al. 2009)
* RandomSequenceTask
* NeuronalAvalanches (from Del Papa et al. 2017, Del Papa et al. 2018)
* LanguageTask (in progress...)

### Prerequisites

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikitlearn](http://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [bunch](https://pypi.python.org/pypi/bunch)
* [powerlaw](https://pypi.python.org/pypi/powerlaw) (for the NeuronalAvalanches experiment, from Del Papa et al. 2017)

### Directory structure

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 2.7](https://www.python.org/download/releases/2.7/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda, not anaconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with "sorn_env") and install the [required packages](https://github.com/delpapa/sorn/blob/master/requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=2.7`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer. Importantly, this SORN repository does not yet support python 3.

## Experiments

Each different experiment has it's own project folder (for example, `CountingTask`), which contains it's parameters, experiment instructions, sources and plot files. If you plan to extend the repository with new experiments, please keep this structure to avoid unecessary conflicts.

### CountingTask

This task is the reimplementation of the counting task from the original SORN paper by [Lazar et al. 2009](http://journal.frontiersin.org/article/10.3389/neuro.10.023.2009/full). The task consist of randomly alternating sequences of the form 'ABB..BBC' and 'DEE..EEF', with size L = n+2.

### RandomSequenceTask

This task implements a random sequence input, which is used to estimate the SORN's fading memory. The random sequence input consists of a sequence of A symbols, and the fading memory capacity is the SORN's capacity of recovering past inputs with it's linear readout layer.

### NeuronalAvalanches

### LanguageTask

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
