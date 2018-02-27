# A Self-Organizing Recurrent Neural Network (SORN)
A SORN repository for general purposes, containing a few experiments and examples. This repository is based on the original [SORN repository](https://github.com/chrhartm/SORN) by Christoph Hartmann combined with adaptations to new experiments.  

Currently under construction... :)

## Getting Started

The scripts for experiment are stored in different folders with their respective name. Each folder contain at least scripts with the experiment parameters ('param.py'), instructions ('experiment.py'), and the input details ('source.py'). The parameters in these scripts can be modified at will to reproduce different results. Additionally, each folder contains the plot scripts, for plotting the results.

To simulate a single experiment, 'python common/run_single.py <ExperimentName>'.

Currently implemented experiments: CountingTask, RandomSequenceTask, NeuronalAvalanches, LanguageTask

### Prerequisites

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [bunch](https://pypi.python.org/pypi/bunch)
* [powerlaw](https://pypi.python.org/pypi/powerlaw) (for the NeuronalAvalanches project, from Del Papa et al. 2017)

### Installing

## Projects

Each different project has it's own project folder (for example, 'CountingTask'), which contain it's parameters, experiment instructions, sources and plot files.

### CountingTask

This task is the reimplementation of the counting task from the original SORN paper by [Lazar et al. 2009](http://journal.frontiersin.org/article/10.3389/neuro.10.023.2009/full).

### RandomSequenceTask

### NeuronalAvalanches

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
