# A Self-Organizing Recurrent Neural Network (SORN)
A SORN repository for general purposes, containing a few experiments and examples. This repository is based on the original [SORN repository](https://github.com/chrhartm/SORN) by Christoph Hartmann combined with adaptations to new experiments.  

Currently under construction... :)

## Getting Started

The scripts for each experiment are stored in different folders with the experiments' respective name. Each folder contain a minimum of three scripts: the experiment parameters ('param.py'), instructions ('experiment.py'), and the input details ('source.py'). The parameters in these scripts can be modified at will to reproduce different results. Additionally, each folder contains the relevant plot scripts, for visualizing the results.

To simulate a single experiment, run 'python common/run_single.py <ExperimentName>' from the main folder.

Currently implemented experiments: CountingTask, RandomSequenceTask, NeuronalAvalanches, LanguageTask

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

Fork this repository onto your own github account and clone it into your favorite SORN folder:

'git clone PATH_TO_YOUR_FAVORITE_SORN_REPOSITORY'

#### Setting up the environment

Install [Python 2.7]() and the conda package manager (use miniconda, not anaconda, because you need to install the packages we need). Navigate to the project directory inside a terminal and create a virtual environment (replace <environment_name>, for example, with "sorn_environment") and install the required packages:

'conda create -n <environment_name> --file requirements.txt'

## Projects

Each different project has it's own project folder (for example, 'CountingTask'), which contain it's parameters, experiment instructions, sources and plot files.

### CountingTask

This task is the reimplementation of the counting task from the original SORN paper by [Lazar et al. 2009](http://journal.frontiersin.org/article/10.3389/neuro.10.023.2009/full). The task consist of randomly alternating sequences of the form 'ABB..BBC' and 'DEE..EEF', with size L = n+2.

* TODO: fix performance calculation (internal state instead of activity)
* TODO: improve 'plot_performance.py' script

### RandomSequenceTask

### NeuronalAvalanches

* TODO: add plot scripts from FIAS PC
* TODO: merge with FIAS PC codes and so on...
* TODO: document properly

### LanguageTask

* TODO: everything

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
