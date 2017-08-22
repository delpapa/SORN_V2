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
            self.total_activity = np.zeros(params.aux.N_steps)

        if 'ConnectionFractionStat' in stats_tosave:
            self.connec_frac = np.zeros(params.aux.N_steps)


        if 'ActivityReadoutStat' in stats_tosave:
            self.total_activity = np.zeros(params.aux.readout_steps)

        if 'CountingLetterStat' in stats_tosave:
            self.letters = np.zeros((params.aux.readout_steps))
            self.sequence_ind = np.zeros((params.aux.readout_steps))

        if 'CountingActivityStat' in stats_tosave:
            self.activity = np.zeros((params.aux.readout_steps, params.par.N_e))

        if 'InternalStateStat' in stats_tosave:
            self.internal_state = np.zeros((params.aux.readout_steps,
                                            params.par.N_e))

    def save_step(self, ):

                    if phase in ['train', 'test']:
                        if phase == 'train':
                            step = n
                        if phase == 'test':
                            step = n + self.params.aux.steps_readouttrain

                        if hasattr(self, 'total_activity'):
                            stats.total_activity[step] = x.sum()
                        if hasattr(self, 'connec_frac'):
                            stats.connec_frac[step] = W_ee.W.sum()
                        if hasattr(self, 'activity'):
                            stats.activity[step] = x
                        if hasattr(self, 'letters'):
                            stats.sequence_ind[step] = int(source.sequence_ind())
                            stats.letters[step] = int(np.argmax(u))
                        if hasattr(self, 'internal_state'):
                            stats.internal_state[step] = x_int
