import numpy as np

################################################################################
# Stats tracker: this file initializes all the possible trackers for every     #
#                experiment. Different experiments can use the same stats      #
#                                                                              #
#   'ActivityStat': stores the total activity (number of active neurons)       #
#   'CountingLetterStat': for the CountingTask                                 #
################################################################################

class Stats(object):
    """Stats to be  store at each simulation step."""
    def __init__(self, stats_tostore, params):

        self.par = params.par
        self.aux = params.aux

        if 'ActivityStat' in stats_tostore:
            self.activity = np.zeros(params.aux.N_steps)

        if 'ActivityReadoutStat' in stats_tostore:
            self.activity_readout = np.zeros(params.aux.readout_steps)

        if 'ConnectionFractionStat' in stats_tostore:
            self.connec_frac = np.zeros(params.aux.N_steps)

        if 'InputReadoutStat' in stats_tostore:
            self.input_readout = np.zeros((params.aux.readout_steps))
            self.input_index_readout = np.zeros((params.aux.readout_steps))

        if 'RasterReadoutStat' in stats_tostore:
            self.raster_readout = np.zeros((params.aux.readout_steps,\
                                            params.par.N_e))

    def store_step(self, x, u, source, W_ee, step, phase):

        # this is necessary to keep sotring data from train and test phases
        # in the same array
        readout = ['train', 'test']
        if phase == 'test':
            step += self.aux.steps_readouttrain

        if hasattr(self, 'activity'):
            self.activity[step] = x.sum()

        if hasattr(self, 'activity_readout') and phase in readout:
            self.activity_readout[step] = x.sum()

        if hasattr(self, 'connec_frac'):
            self.connec_frac[step] = W_ee.W.data.size / float(self.par.N_e**2)

        if hasattr(self, 'input_readout') and phase in readout:
            self.input_readout[step] = int(np.argmax(u))
            self.input_index_readout[step] = int(source.sequence_ind())

        if hasattr(self, 'raster_readout') and phase in readout:
            self.raster_readout[step] = x
