""" Counting Task experiment

This script contains the experimental instructions for the Counting Task
experiment.
"""

import copy

from sklearn import linear_model

from .source import CountingSource

class Experiment:
    """
    Experiment for the SORN: contains the source, the simulation procedure and
    the performance calculation
    """
    def __init__(self, params):

        # always keep track of initial sorn parameters
        self.init_params = copy.deepcopy(params.par)

        # results directory name
        self.results_dir = (params.aux.experiment_name
                            + params.aux.experiment_tag
                            + '/N' + str(params.par.N_e)
                            + '_L' + str(params.par.L))

        # define which stats to store during the simulation
        self.stats_cache = [
            'InputReadoutStat',
            'RasterReadoutStat',
        ]

        # define which parameters and files to save at the end of the simulation
        #     params: save initial main sorn parameters
        #     stats: save all stats trackers
        #     scripts: backup scripts used during the simulation
        self.files_tosave = [
            'params',
            'stats',
            'scripts',
        ]

        # load input source
        self.inputsource = CountingSource(self.init_params)

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
            print('Plasticity phase:')
        sorn.simulation(stats, phase='plastic')

        # 2. input without plasticity - train (STDP and IP off)
        if display:
            print('\nReadout training phase:')

        sorn.params.par.eta_stdp = 'off'
        sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')

        # 3. input without plasticity - test performance (STDP and IP off)
        if display:
            print('\nReadout testing phase:')

        sorn.simulation(stats, phase='test')

        # 4. calculate performance
        if display:
            print('\nCalculating performance... ', end='')

        # load stats to calculate the performance
        t_train = sorn.params.aux.steps_readouttrain
        t_test = sorn.params.aux.steps_readouttest

        X_train = stats.raster_readout[:t_train-1].T
        # y_train = (stats.input_readout[1:t_train].T).astype(int)
        y_train_ind = (stats.input_index_readout[1:t_train].T).astype(int)

        X_test = stats.raster_readout[t_train:t_train+t_test-1].T
        # y_test = (stats.input_readout[t_train+1:t_train+t_test].T).astype(int)
        y_test_ind = (stats.input_index_readout[t_train+1:t_train+t_test].T).astype(int)

        # Logistic Regression
        readout = linear_model.LogisticRegression(multi_class='multinomial',
                                                  solver='lbfgs')
        output_weights = readout.fit(X_train.T, y_train_ind)
        performance = output_weights.score(X_test.T, y_test_ind)

        # #### Readout training using PI matrix
        # # trasform labels in one-hot array
        # onehot_values = np.max(y_train) + 1
        # y_train_onehot = np.eye(onehot_values)[y_train_ind]
        # y_test_onehot = np.eye(onehot_values)[y_test_ind]
        #
        # X_train_pinv = np.linalg.pinv(X_train) # MP pseudo-inverse
        # W_trained = np.dot(y_train_onehot.T, X_train_pinv) # least squares
        #
        # # Network prediction with trained weights
        # y_predicted = np.dot(W_trained, X_test)
        #
        # # Performance by Pseudo-Inverse
        # prediction = np.argmax(y_predicted, axis=0)
        # performance_PI = (prediction == y_test).sum()/float(len(y_test))

        stats.performance = performance

        if display:
            print('done')
