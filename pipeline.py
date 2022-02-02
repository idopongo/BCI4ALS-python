import mne
from Marker import Marker
import os
import json
from preprocessing import Preprocessor
from features import get_features, FeatureExtractor
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
    raw, params = load_recordings("Haggai")
    epochs, labels = get_epochs(raw, params["trial_duration"])
    best_hyperparams_haggai = {
        "features__n_fft": 200,
        "features__n_per_seg": 62,
        "features__n_overlap": 0.3,
        "features__freq_bands": [8, 12, 30],
        "preprocessor__trim_epoch": 0,
        "preprocessor__l_freq": 5,
        "preprocessor__h_freq": 30,
    }
    best_hyperparams_ido = {
        "features__n_fft": 250,
        "features__n_per_seg": 62,
        "features__n_overlap": 0.2,
        "features__freq_bands": [8, 12, 20],
        "preprocessor__trim_epoch": 1,
        "preprocessor__l_freq": 2,
        "preprocessor__h_freq": 24,
    }
    save_hyperparams(best_hyperparams_ido, "Ido")
    pipeline = create_pipeline(best_hyperparams_haggai)
    skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
    scores = cross_val_score(pipeline, epochs.get_data(), labels, cv=skf)
    print(
        f'features classifier accuracy: \n mean: {np.round(np.mean(scores), 2)} \n std: {np.round(np.std(scores), 3)}')
    # grid_search_pipeline(epochs, labels)


def create_pipeline(hyperparams):
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline([('preprocessor', Preprocessor()), ('features', FeatureExtractor()), ('lda', lda)])
    for name, step in pipeline.named_steps.items():
        step_params = [v for k, v in hyperparams.items() if k.startswith(name)]
        step.set_params(*step_params)
    return pipeline


def save_hyperparams(hyperparams, name):
    filename = f'{name}_hyperparams.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)


def grid_search_pipeline(epochs, labels):
    lda = LinearDiscriminantAnalysis()

    pipeline = Pipeline([('preprocessor', Preprocessor()), ('features', FeatureExtractor()), ('lda', lda)])
    gridsearch_params = {
        "features__n_fft": [150, 200, 250],
        "features__n_per_seg": [31, 62, 125],
        "features__n_overlap": [0.2, 0.3, 0.4],
        "features__freq_bands": [[8, 12, 30], [8, 12, 20, 30]],
        "preprocessor__trim_epoch": [0, 1],
        "preprocessor__l_freq": [2, 5, 8],
        "preprocessor__h_freq": [24, 30],
    }
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)

    gs = GridSearchCV(pipeline, gridsearch_params, cv=skf)
    gs.fit(epochs.get_data(), labels)
    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
    print(gs.best_params_)


def get_epochs(raw, trial_duration):
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, Marker.all(), tmin=-1, tmax=trial_duration, picks="data", baseline=(-1, 0))
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


def save_pipeline(name, classifier_num, clf):
    str_classifier = "classifier_{}_{}.pickle".format(name, classifier_num)
    classifier_filename = os.path.join(PIPELINES_DIR, str_classifier)
    pickle_write = open(classifier_filename, "wb")
    pickle.dump(clf, pickle_write)
    pickle_write.close()


if __name__ == "__main__":
    main()
