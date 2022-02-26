import mne
from ..recording.Marker import Marker
from preprocessing import Preprocessor, reject_epochs
from features import FeatureExtractor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from ..data_utils import load_recordings

DEFAULT_HYPERPARAMS = {
    "preprocessing__epoch_tmin": 0.1,
    "preprocessing__l_freq": 2,
    "preprocessing__h_freq": 24,
    "feature_extraction__n_fft": 250,
    "feature_extraction__n_per_seg": 120,
    "feature_extraction__n_overlap": 0,
    "feature_extraction__freq_bands": [8, 12, 30],
}


def evaluate_pipeline(pipeline, epochs, labels):
    skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10)
    scores = cross_val_score(pipeline, epochs, labels, cv=skf)
    print(
        f'features classifier accuracy: \n mean: {np.round(np.mean(scores), 2)} \n std: {np.round(np.std(scores), 3)}')


def create_and_fit_pipeline(raw, recording_params, hyperparams=DEFAULT_HYPERPARAMS):
    # get data, epochs
    epochs, labels = get_epochs(raw, recording_params["trial_duration"],
                                reject_bad=not recording_params['use_synthetic_board'])
    # create a pipeline from params
    pipeline = create_csp_pipeline(hyperparams)
    pipeline.fit(epochs, labels)
    return pipeline, epochs, labels


def create_pipeline(hyperparams=DEFAULT_HYPERPARAMS):
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline([('preprocessing', Preprocessor()), ('feature_extraction', FeatureExtractor()), ('lda', lda)])
    hyperparams = {key: hyperparams[key] for key in hyperparams if key.split("__")[0] in pipeline.get_params().keys()}
    pipeline.set_params(**hyperparams)
    return pipeline


def create_csp_pipeline(hyperparams=DEFAULT_HYPERPARAMS):
    lda = LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP()
    pipeline = Pipeline([('preprocessing', Preprocessor()), ('CSP', csp), ('lda', lda)])
    hyperparams = {key: hyperparams[key] for key in hyperparams if key.split("__")[0] in pipeline.get_params().keys()}
    pipeline.set_params(**hyperparams)
    return pipeline


def grid_search_pipeline_hyperparams(epochs, labels):
    pipeline = create_csp_pipeline()
    gridsearch_params = {
        "preprocessing__epoch_tmin": [0.1, 1],
        "preprocessing__l_freq": [2, 5, 7],
        "preprocessing__h_freq": [24, 28],
        "feature_extraction__n_fft": [250],
        "feature_extraction__n_per_seg": [120],
        "feature_extraction__n_overlap": [0],
        "feature_extraction__freq_bands": [[11, 30], [8, 12, 30]],
    }
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    mne.set_log_level(verbose="WARNING")
    gs = GridSearchCV(pipeline, gridsearch_params, cv=skf, n_jobs=-1, verbose=10, error_score="raise")
    mne.set_log_level(verbose="INFO")
    gs.fit(epochs, labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)
    return gs.best_params_


def grid_search_csp_hyperparams(epochs, labels):
    pipeline = create_pipeline()
    gridsearch_params = {
        "preprocessing__epoch_tmin": [0, 0.5, 1, 1.5],
        "preprocessing__l_freq": [5, 7],
        "preprocessing__h_freq": [27, 30],
        "CSP__n_components": [5, 6, 7, 8]
    }
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    mne.set_log_level(verbose="WARNING")
    gs = GridSearchCV(pipeline, gridsearch_params, cv=skf, n_jobs=-1, verbose=10, error_score="raise")
    mne.set_log_level(verbose="INFO")
    gs.fit(epochs, labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)
    return gs.best_params_


def get_epochs(raw, trial_duration, markers=Marker.all(), reject_bad=False):
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, markers, tmin=-1, tmax=trial_duration, picks="data", baseline=(-1, 0))

    # running get data triggers dropping of epochs, we want to make sure this happens now so that the labels are
    # consistent with the epochs
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1]

    if reject_bad:
        epochs_data, labels = reject_epochs(epochs_data, labels)

    return epochs_data, labels


if __name__ == "__main__":
    raw, params = load_recordings("Synthetic")
    create_and_fit_pipeline(raw, params)
