import mne
from Marker import Marker
import os
import json
from preprocessing import Preprocessor
from features import FeatureExtractor
from classifier import create_classifier
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from constants import *
from sklearn.model_selection import GridSearchCV
import pickle


def main():
    subject = "David3"
    raw, params = load_recordings(subject)
    epochs, labels = get_epochs(raw, params["trial_duration"])
    epochs = epochs.get_data()
    best_params = grid_search_pipeline_hyperparams(epochs, labels)
    save_hyperparams(best_params, subject)
    pipeline = create_pipeline(best_params)
    pipeline.fit(epochs, labels)
    save_pipeline(pipeline, subject)
    skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10)
    scores = cross_val_score(pipeline, epochs, labels, cv=skf)
    print(
        f'features classifier accuracy: \n mean: {np.round(np.mean(scores), 2)} \n std: {np.round(np.std(scores), 3)}')


def create_pipeline(hyperparams={}):
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


def save_hyperparams(hyperparams, name):
    filename = f'{name}_hyperparams.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)


def grid_search_pipeline_hyperparams(epochs, labels):
    pipeline = create_pipeline()
    gridsearch_params = {
        "preprocessing__epoch_tmin": [0, 1],
        "preprocessing__l_freq": [2, 5, 8],
        "preprocessing__h_freq": [24, 30],
        "feature_extraction__n_fft": [150, 200, 250],
        "feature_extraction__n_per_seg": [31, 62, 125],
        "feature_extraction__n_overlap": [0.2, 0.3, 0.4],
        "feature_extraction__freq_bands": [[8, 12, 30], [8, 12, 20, 30]],
    }
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)

    gs = GridSearchCV(pipeline, gridsearch_params, cv=skf, n_jobs=-1)
    gs.fit(epochs, labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)
    return gs.best_params_


def get_epochs(raw, trial_duration, markers=Marker.all()):
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, markers, tmin=-1, tmax=trial_duration, picks="data", baseline=(-1, 0))
    labels = epochs.events[:, -1]
    return epochs, labels


def load_recordings(subj):
    subj_recs = get_subject_rec_folders(subj)
    raws = [mne.io.read_raw_fif(os.path.join(RECORDINGS_DIR, rec, 'raw.fif')) for rec in subj_recs]
    with open(os.path.join(RECORDINGS_DIR, subj_recs[0], 'params.json')) as file:
        params = json.load(file)
    all_raw = mne.io.concatenate_raws(raws)
    print(subj_recs)
    return all_raw, params


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
    main()
