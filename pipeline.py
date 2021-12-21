import mne
from Marker import Marker
from constants import *
import os
import json
from preprocessing import preprocess
from features import get_features
from classifier import create_classifier

def main():
    raw, params = load_recordings("David3")
    raw.load_data()
    raw = preprocess(raw)
    epochs, labels = get_epochs(raw, params["trial_duration"])
    features = get_features(epochs.get_data())
    clf, acc = create_classifier(features, labels)
    print(f'k-fold validation accuracy: {acc}')

def get_epochs(raw, trial_duration):
    events = mne.find_events(raw)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), 0, trial_duration, picks="data", baseline=(0, 0))
    labels = epochs.events[:, -1]
    return epochs, labels


def load_recordings(subj):
    recs = os.listdir(RECORDINGS_DIR)
    subj_recs = [rec for rec in recs if rec.split("_")[1] == subj]
    raws = [mne.io.read_raw_fif(os.path.join(RECORDINGS_DIR, rec, 'raw.fif')) for rec in recs]
    with open(os.path.join(RECORDINGS_DIR, subj_recs[0], 'params.json')) as file:
        params = json.load(file)
    all_raw = mne.io.concatenate_raws(raws)
    return all_raw, params


if __name__ == "__main__":
    main()
