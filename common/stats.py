import numpy as np

################################################################################
# Stats tracker: this file initializes all the possible trackers for every     #
#                experiment. Different experiments can use the same stats      #
#                                                                              #
#   'ActivityStat': stores the total activity (number of active neurons)       #
#   'CountingLetterStat': for the CountingTask                                 #
################################################################################

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
