import numpy as np

import utils
stats = utils.Bunch()

class Stats(object):
    """
    Stats to be saved at the end of the simulation
    """

    def __init__(self, stats_tosave, params):

        if 'ActivityStat' in stats_tosave:
            self.total_activity = np.zeros(params.readout_steps)

        if 'CountingLetterStat' in stats_tosave:
            self.letters = np.zeros((params.readout_steps))
            self.sequence_ind = np.zeros((params.readout_steps))

        if 'CountingActivityStat' in stats_tosave:
            self.activity = np.zeros((params.readout_steps, params.N_e))

        if 'ConnectionFractionStat' in stats_tosave:
            self.connec_frac = np.zeros(params.N_steps)

        if 'InternalStateStat' in stats_tosave:
            self.internal_state = np.zeros((params.readout_steps, params.N_e))
