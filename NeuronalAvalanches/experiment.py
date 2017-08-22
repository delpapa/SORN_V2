import random as randomstr
import copy

import numpy as np
import sklearn
from sklearn import linear_model

from source import NoSource

class Experiment(object):
    """
    Experiment for the SORN: contains the source, the simulation procedure and
    the performance calculation
    """
    def __init__(self, params):

            # keep track of initial parameters
            self.init_params = copy.deepcopy(params.par)

            # define results dir name
            self.results_dir = (params.aux.experiment_name
                                + params.aux.experiment_tag
                                + '/N' + str(params.par.N_e)
                                + '_sigma' + str(params.par.sigma))

    def start(self):
        """
        Start the experiment
        """
        self.stats_tosave = [
                             'ActivityStat',
                             # 'CountingLetterStat',
                             # 'CountingActivityStat',
                             'ConnectionFractionStat'
                             # 'InternalStateStat'
                            ]

        self.files_tosave = [
                             'params',
                             'stats',
                             # 'scripts'
                            ]

        self.inputsource =  NoSource(self.init_params)

    def run(self, sorn, stats):
        """
        Run experiment once

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
            print 'done'
