import mne
from Marker import Marker
from constants import *
import os
import json
from preprocessing import preprocess
from features import get_features
from classifier import create_classifier
import numpy as np
import scipy.io
import pickle


def main():
    name = 'Haggai'
    raw, params = load_recordings(name)
    raw = preprocess(raw)
    epochs, labels = get_epochs(raw, params["trial_duration"])
    features = get_features(epochs.get_data())
    clf, acc = create_classifier(features, labels)
    classifier_num = 1
    str_classifier = "classifier_{}_{}.pickle".format(name, classifier_num)
    classifier_filename = os.path.join(CLASSIFIERS_DIR, str_classifier)
    pickle_write = open(classifier_filename, "wb")
    pickle.dump(clf, pickle_write)
    pickle_write.close()
    print(f'k-fold validation accuracy: {np.mean(acc)}')


def get_epochs(raw, trial_duration):
    events = mne.find_events(raw)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), tmin=0, tmax=trial_duration, picks="data", baseline=(0, 0))
    labels = epochs.events[:, -1]
    return epochs, labels


def load_recordings(subj):
    recs = os.listdir(RECORDINGS_DIR)
    subj_recs = [rec for rec in recs if rec.split("_")[1] == subj]
    raws = [mne.io.read_raw_fif(os.path.join(RECORDINGS_DIR, rec, 'raw.fif')) for rec in subj_recs]
    with open(os.path.join(RECORDINGS_DIR, subj_recs[0], 'params.json')) as file:
        params = json.load(file)
    all_raw = mne.io.concatenate_raws(raws)
    return all_raw, params


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
