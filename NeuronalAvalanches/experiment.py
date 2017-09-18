import random as randomstr
import copy

import numpy as np
import sklearn
from sklearn import linear_model

from source import NoSource

class Experiment(object):
    """Experiment class.

    It contains the source, the simulation procedure and back up instructions.
    """
    def __init__(self, params):
        """Start the experiment.

        Initialize relevant variables and stats trackers.

        Parameters:
            params: Bunch
                All sorn inital parameters
        """
        # always keep track of initial sorn parameters
        self.init_params = copy.deepcopy(params.par)

        # results directory name
        self.results_dir = (params.aux.experiment_name
                            + params.aux.experiment_tag
                            + '/N' + str(params.par.N_e)
                            + '_sigma' + str(params.par.sigma))

        # define which stats to store during the simulation
        self.stats_tostore = [
                             'ActivityStat',
                             'ConnectionFractionStat'
                             ]

        # define which parameters and files to save at the end of the simulation
        #     params: save initial main sorn parameters
        #     stats: save all stats trackers
        #     scripts: backup scripts used during the simulation
        self.files_tosave = [
                             'params',
                             'stats',
                             # 'scripts'
                            ]

        # load input source
        self.inputsource = NoSource(self.init_params)

    def run(self, sorn, stats):
        """Run experiment once.

        Parameters:
            sorn: Bunch
                The bunch of sorn parameters
            stats: Bunch
                The bunch of stats to save
        """
        display = sorn.params.aux.display

        # 1. input with plasticity
        if display:
            print 'Plasticity phase:'
        sorn.simulation(stats, phase='plastic')
        if display:
            print '... done'
