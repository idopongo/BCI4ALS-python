from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preprocessing import laplacian
import mne
from mne_features.univariate import compute_pow_freq_bands
import numpy as np
from pipeline import show_pipeline_steps, filter_hyperparams_for_pipeline
from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


class Preprocessor:
    def __init__(self):
        self.l_freq = 7
        self.h_freq = 30
        self.do_laplacian = True

    def set_params(self, l_freq=None, h_freq=None, do_laplacian=None):
        if l_freq is not None:
            self.l_freq = l_freq
        if h_freq is not None:
            self.h_freq = h_freq
        if do_laplacian is not None:
            self.do_laplacian = do_laplacian

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq, verbose=False)
        if self.do_laplacian:
            epochs = laplacian(epochs)
        return epochs


class FeatureExtractor:
    def __init__(self):
        self.epoch_tmin = 0
        self.freq_bands = [8, 12, 30]
        self.n_per_seg = 100

    def set_params(self, epoch_tmin, n_per_seg=None, freq_bands=None):
        if n_per_seg is not None:
            self.n_per_seg = n_per_seg
        if epoch_tmin is not None:
            self.epoch_tmin = epoch_tmin
        if freq_bands is not None:
            self.freq_bands = freq_bands

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        psd_params = {"welch_n_fft": 512, "welch_n_per_seg": self.n_per_seg}
        sfreq = 125
        tmin = self.epoch_tmin
        imagination_start = int(tmin * sfreq)
        band_power = np.array(
            [compute_pow_freq_bands(sfreq, epoch[:, imagination_start:], self.freq_bands, normalize=True,
                                    psd_params=psd_params) for
             epoch in
             epochs]
        )
        band_power_calib = np.array(
            [compute_pow_freq_bands(sfreq, epoch[:, :imagination_start], self.freq_bands, normalize=True,
                                    psd_params=psd_params) for
             epoch in
             epochs]
        )
        sre = band_power_calib / band_power
        features = np.concatenate((band_power, sre), axis=1)
        return features


class CategoricalList(Categorical):
    def __init__(self, categories, **categorical_kwargs):
        super().__init__(self._convert_hashable(categories), **categorical_kwargs)

    def _convert_hashable(self, list_of_lists):
        return [self._HashableListAsDict(list_)
                for list_ in list_of_lists]

    class _HashableListAsDict(dict):
        def __init__(self, arr):
            self.update({i: val for i, val in enumerate(arr)})

        def __hash__(self):
            return hash(tuple(sorted(self.items())))

        def __repr__(self):
            return str(list(self.values()))

        def __getitem__(self, key):
            return list(self.values())[key]


bayesian_search_space = {
    # "preprocessing__l_freq": [2],
    # "preprocessing__h_freq": [40],
    # "preprocessing__do_laplacian": [True, False],
    # "feature_extraction__epoch_tmin": Real(1, 2),
    # "feature_extraction__freq_bands": Categorica
    # lList([[7, 12, 30]]),
    # "feature_extraction__n_per_seg": [230],
    "model__max_features": Integer(8, 30),
    "model__n_estimators": Integer(1, 1000),
    # "model__max_depth": Integer(1, 6),
    # 'model__learning_rate': Real(0.0005, 0.9, prior="log-uniform"),
}

grid_search_space = {
    "preprocessing__l_freq": [3, 5, 7, 10],
    "preprocessing__h_freq": [20, 24, 28, 30],
    "feature_extraction__n_per_seg": [375 / 3, 375 / 5, 375 / 7],
}


def create_pipeline(hyperparams=None):
    default_hyperparams = {
        "preprocessing__l_freq": 2,
        "preprocessing__h_freq": 40,
        "preprocessing__do_laplacian": False,
        "feature_extraction__epoch_tmin": 1.7,
        "feature_extraction__n_per_seg": 230,
        "feature_extraction__freq_bands": [7, 12, 30],
        # "model__n_estimators": 50,
        # 'model__learning_rate': 0.01,
        # 'model__loss': 'deviance'
    }

    if hyperparams:
        hyperparams = {**default_hyperparams, **hyperparams}
    else:
        hyperparams = default_hyperparams

    model = RandomForestClassifier()
    pipeline = Pipeline(
        [('preprocessing', Preprocessor()), ('feature_extraction', FeatureExtractor()), ('model', model)])
    hyperparams = filter_hyperparams_for_pipeline(hyperparams, pipeline)
    pipeline.set_params(**hyperparams)
    print(f'Creating spectral pipeline: {show_pipeline_steps(pipeline)}')

    return pipeline
