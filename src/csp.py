from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preprocessing import laplacian
import mne
import numpy as np
from pipeline import show_pipeline_steps
from mne.decoding import CSP
from skopt.space import Real


class Preprocessor:
    def __init__(self):
        self.epoch_tmin = 1
        self.l_freq = 7
        self.h_freq = 30
        self.do_laplacian = True

    def set_params(self, epoch_tmin, l_freq, h_freq, do_laplacian):
        self.epoch_tmin = epoch_tmin
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.do_laplacian = do_laplacian

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq, verbose=False)
        epochs = epochs[:, :, int(125 * self.epoch_tmin):]
        if self.do_laplacian:
            epochs = laplacian(epochs)
        return epochs


bayesian_search_space = {
    "preprocessing__epoch_tmin": Real(0, 2),
    "preprocessing__l_freq": Real(2, 14),
    "preprocessing__h_freq": Real(15, 35),
    "preprocessing__do_laplacian": [True, False],
    "csp__n_components": [9, 10, 11, 12, 13],
}

grid_search_space = {
    "preprocessing__epoch_tmin": [0, 0.2, 0.5, 0.7, 1, 1.5],
    "preprocessing__l_freq": [3, 5, 7, 10],
    "preprocessing__h_freq": [20, 24, 28, 30],
    "CSP__n_components": [9, 10, 11, 12, 13, 14],
}


def create_pipeline(hyperparams=None):
    default_hyperparams = {
        "preprocessing__epoch_tmin": 0,
        "preprocessing__l_freq": 8.3,
        "preprocessing__h_freq": 36,
        "preprocessing__do_laplacian": True,
        "csp__n_components": 11,
    }
    if hyperparams:
        hyperparams = {**default_hyperparams, **hyperparams}
    else:
        hyperparams = default_hyperparams
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline(
        [('preprocessing', Preprocessor()), ('csp', CSP(log=True)), ('lda', lda)])
    pipeline.set_params(**hyperparams)
    print(f'Creating CSP pipeline: {show_pipeline_steps(pipeline)}')

    return pipeline
