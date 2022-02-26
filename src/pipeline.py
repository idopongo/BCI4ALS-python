import mne
from Marker import Marker
from preprocessing import Preprocessor, reject_epochs
from features import FeatureExtractor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from src.data_utils import load_recordings

mne.set_log_level('warning')

DEFAULT_HYPERPARAMS = {
    "preprocessing__epoch_tmin": 0.1,
    "preprocessing__l_freq": 2,
    "preprocessing__h_freq": 24,
    "feature_extraction__n_fft": 125,
    "feature_extraction__n_per_seg": 120,
    "feature_extraction__n_overlap": 0.3,
    "feature_extraction__freq_bands": [8, 12, 30],
    "CSP__n_components": 5,
}


def evaluate_pipeline(pipeline, epochs, labels):
    n_splits = 3
    n_repeats = 10
    print(f'Evaluating pipeline performance ({n_splits} splits, {n_repeats} repeats)...')
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    scores = cross_val_score(pipeline, epochs, labels, cv=skf)
    print(
        f'Accuracy: \n mean: {np.round(np.mean(scores), 2)} \n std: {np.round(np.std(scores), 3)}')


def create_and_fit_pipeline(raw, recording_params, hyperparams=DEFAULT_HYPERPARAMS,
                            pipeline_type="spectral"):
    # get data, epochs
    epochs, labels = get_epochs(raw, recording_params["trial_duration"],
                                reject_bad=not recording_params['use_synthetic_board'])
    # create a pipeline from params
    pipeline = create_pipeline(hyperparams, pipeline_type=pipeline_type)
    pipeline.fit(epochs, labels)
    return pipeline, epochs, labels


def create_pipeline(hyperparams=DEFAULT_HYPERPARAMS, pipeline_type="spectral"):
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS
    lda = LinearDiscriminantAnalysis()
    if pipeline_type == "spectral":
        pipeline = Pipeline(
            [('preprocessing', Preprocessor()), ('feature_extraction', FeatureExtractor()), ('lda', lda)])
    elif pipeline_type == "csp":
        pipeline = Pipeline([('preprocessing', Preprocessor()), ('CSP', mne.decoding.CSP()), ('lda', lda)])
    else:
        raise ValueError(f'Pipeline type {pipeline_type} is not supported')

    print(f'Creating {pipeline_type} pipeline: {show_pipeline_steps(pipeline)}')
    relevant_hyperparams = filter_hyperparams_for_pipeline(hyperparams, pipeline)
    pipeline.set_params(**relevant_hyperparams)
    return pipeline


def filter_hyperparams_for_pipeline(hyperparams, pipeline):
    return {key: hyperparams[key] for key in hyperparams if
            key.split("__")[0] in pipeline.get_params().keys()}


def show_pipeline_steps(pipeline):
    return " => ".join(list(pipeline.named_steps.keys()))


def grid_search_pipeline_hyperparams(epochs, labels, pipeline_type):
    pipeline = create_pipeline(pipeline_type=pipeline_type)
    gridsearch_params = {
        "preprocessing__epoch_tmin": [0.1, 1],
        "preprocessing__l_freq": [2, 5, 7],
        "preprocessing__h_freq": [24, 28],
        "feature_extraction__n_fft": [250],
        "feature_extraction__n_per_seg": [120],
        "feature_extraction__n_overlap": [0],
        "feature_extraction__freq_bands": [[11, 30], [8, 12, 30]],
        "CSP__n_components": [5, 6, 7, 8]
    }
    gridsearch_params = filter_hyperparams_for_pipeline(gridsearch_params, pipeline)
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
    print(f'Found {len(labels)} epochs')

    if reject_bad:
        epochs_data, labels = reject_epochs(epochs_data, labels)

    return epochs_data, labels


if __name__ == "__main__":
    raw, params = load_recordings("Synthetic")
    create_and_fit_pipeline(raw, params)
