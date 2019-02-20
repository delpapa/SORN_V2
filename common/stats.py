"""Stats tracker

This script initializes all the possible trackers for each simulation.
Different experiments may use the same stats.

   'ActivityStat': the total activity (number of active neurons)
   'ActivityReadoutStat': the total activity only for the readout
   'ConnectionFractionStat': the fraction of active E-E connections
   'InputReadoutStat': the input and input index for the readout
   'RasterReadoutStat': the raster (activity of each neuron) for the readout
"""

import numpy as np


class Stats(object):
    """
    Stats to be store at each simulation step.
    """

    def __init__(self, stats_tostore, params):

        self.par = params.par
        self.aux = params.aux

        if 'ActivityStat' in stats_tostore:
            self.activity = np.zeros(params.aux.N_steps, dtype=np.int)

        if 'ActivityReadoutStat' in stats_tostore:
            self.activity_readout = np.zeros(params.aux.readout_steps,
                                             dtype=np.int16)

        if 'ConnectionFractionStat' in stats_tostore:
            self.connec_frac = np.zeros(params.aux.N_steps)

        if 'InputReadoutStat' in stats_tostore:
            self.input_readout = np.zeros(params.aux.readout_steps,
                                          dtype=np.int16)
            self.input_index_readout = np.zeros(params.aux.readout_steps,
                                          dtype=np.int16)

        if 'RasterReadoutStat' in stats_tostore:
            self.raster_readout = np.zeros((params.aux.readout_steps,
                                            params.par.N_e),
                                            dtype=np.int8)

    def store_step(self, x, u, source, W_ee, step, phase):
        """
        Store the stats each time step into numpy arrays.

        Arguments:
        x -- current activity array
        u -- one-hot array of the current input
        source -- the SORN source object
        W_ee -- Sparse synapses matrix of size (N_e, N_e)
        step -- current time step
        phase -- string of the simulation phase
        """

        # this is necessary to keep caching data from train and test phases
        # in the same array
        readout = ['train', 'test']
        if phase == 'test':
            step += self.aux.steps_readouttrain

        if hasattr(self, 'activity'):
            if phase is 'plastic':
                self.activity[step] = x.sum()
            elif phase in readout:
                self.activity[self.par.steps_plastic+step] = x.sum()

        if hasattr(self, 'activity_readout') and phase in readout:
            self.activity_readout[step] = x.sum()

        if hasattr(self, 'connec_frac'):
            if phase is 'plastic':
                self.connec_frac[step] = \
                                       W_ee.W.data.size / float(self.par.N_e**2)
            elif phase in readout:
                self.connec_frac[self.par.steps_plastic+step] = \
                                       W_ee.W.data.size / float(self.par.N_e**2)

        if hasattr(self, 'input_readout') and phase in readout:
            self.input_readout[step] = np.argmax(u)
            self.input_index_readout[step] = source.sequence_ind()

        if hasattr(self, 'raster_readout') and phase in readout:
            self.raster_readout[step] = x
