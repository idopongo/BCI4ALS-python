import mne
from Marker import Marker
import os
import json
from preprocessing import Preprocessor, reject_epochs
from features import FeatureExtractor
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from constants import *
from sklearn.model_selection import GridSearchCV
import pickle

DEFAULT_HYPERPARAMS = {
    "preprocessing__epoch_tmin": 1,
    "preprocessing__l_freq": 2,
    "preprocessing__h_freq": 24,
    "feature_extraction__n_fft": 250,
    "feature_extraction__n_per_seg": 62,
    "feature_extraction__n_overlap": 0.2,
    "feature_extraction__freq_bands": [
        8,
        12,
        20,
        30
    ],
}


def evaluate_pipeline(pipeline, epochs, labels):
    skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10)
    scores = cross_val_score(pipeline, epochs.get_data(), labels, cv=skf)
    print(
        f'features classifier accuracy: \n mean: {np.round(np.mean(scores), 2)} \n std: {np.round(np.std(scores), 3)}')


def create_and_fit_pipeline(raw, recording_params, hyperparams=DEFAULT_HYPERPARAMS):
    # get data, epochs
    epochs, labels = get_epochs(raw, recording_params["trial_duration"])
    epochs, labels = reject_epochs(epochs, labels)

    # create a pipeline from params
    pipeline = create_pipeline(hyperparams)
    pipeline.fit(epochs.get_data(), labels)
    return pipeline, epochs, labels


def create_pipeline(hyperparams=DEFAULT_HYPERPARAMS):
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline([('preprocessing', Preprocessor()), ('feature_extraction', FeatureExtractor()), ('lda', lda)])
    pipeline.set_params(**hyperparams)
    return pipeline


def save_pipeline(pipeline, name):
    save_path = os.path.join(PIPELINES_DIR, f'pipeline_{name}.pickle')
    with open(save_path, "wb") as file:
        pickle.dump(pipeline, file)


def load_pipeline(name):
    load_path = os.path.join(PIPELINES_DIR, f'pipeline_{name}.pickle')
    with open(load_path, "rb") as file:
        pipeline = pickle.load(file)
    return pipeline


def save_hyperparams(hyperparams, subject):
    filename = os.path.join(PIPELINES_DIR, f'{subject}_hyperparams.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)


def load_hyperparams(subject):
    with open(os.path.join(PIPELINES_DIR, f'{subject}_hyperparams.json')) as file:
        hyperparams = json.load(file)
    return hyperparams


def grid_search_pipeline_hyperparams(epochs, labels):
    pipeline = create_pipeline()
    gridsearch_params = {
        "preprocessing__epoch_tmin": [0, 0.5, 1],
        "preprocessing__l_freq": [2, 5, 8],
        "preprocessing__h_freq": [24, 30],
        "feature_extraction__n_fft": [150, 200, 250],
        "feature_extraction__n_per_seg": [31, 62, 125],
        "feature_extraction__n_overlap": [0.2, 0.3, 0.4],
        "feature_extraction__freq_bands": [[8, 12, 30], [8, 12, 20, 30]],
    }
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    mne.set_log_level(verbose="WARNING")
    gs = GridSearchCV(pipeline, gridsearch_params, cv=skf, n_jobs=-1, verbose=10)
    mne.set_log_level(verbose="INFO")
    gs.fit(epochs.get_data(), labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)
    return gs.best_params_


def get_epochs(raw, trial_duration, markers=Marker.all()):
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, markers, tmin=-1, tmax=trial_duration, picks="data", baseline=(-1, 0))

    # running get data triggers dropping of epochs, we want to make sure this happens now so that the labels are
    # consistent with the epochs
    epochs.get_data()

    labels = epochs.events[:, -1]
    return epochs, labels


def load_recordings(subj):
    subj_recs = get_subject_rec_folders(subj)
    raws = [mne.io.read_raw_fif(os.path.join(RECORDINGS_DIR, rec, 'raw.fif')) for rec in subj_recs]
    # When multiple recordings are loaded, the recording_params.json is taken from the first recording
    with open(os.path.join(RECORDINGS_DIR, subj_recs[0], 'params.json')) as file:
        rec_params = json.load(file)
    all_raw = mne.io.concatenate_raws(raws)
    return all_raw, rec_params


def get_subject_rec_folders(subj):
    recs = os.listdir(RECORDINGS_DIR)
    subj_recs = [rec for rec in recs if rec.split("_")[1] == subj]
    return subj_recs


def matlab_data_pipeline():
    data = scipy.io.loadmat("recordings/matlab_data/EEG.mat")["EEG"]
    data = np.moveaxis(data, [0, 1, 2], [1, 2, 0])
    labels = scipy.io.loadmat("recordings/matlab_data/trainingVec.mat")["trainingVec"][0]
    ch_names = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Fz']
    ch_types = ['eeg'] * 16
    sampling_freq = 500
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    events = np.column_stack((np.arange(0, len(labels) * 500, sampling_freq),
                              np.zeros(len(labels), dtype=int),
                              labels))
    epochs = mne.EpochsArray(data, info, events=events)
    features = get_features(epochs.get_data())
    clf, acc = create_classifier(features, labels)
    print(f'k-fold validation accuracy: {np.mean(acc)}')


if __name__ == "__main__":
    raw, params = load_recordings("Synthetic")
    create_and_fit_pipeline(raw, params)
