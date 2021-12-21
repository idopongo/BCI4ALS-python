import mne
from Marker import Marker
from constants import *
import os
import json
from features import get_features
from classifier import create_classifier
import matplotlib

matplotlib.use('Qt5Agg')

def main():
    raw, params = load_recordings("David")
    raw.pick(EEG_CHANNELS[:12] + [16])
    raw.load_data()
    raw.notch_filter(50, notch_widths=2)
    raw.filter(2, 30)
    raw.plot()
    epochs, labels = get_epochs(raw, params["trial_duration"])
    features = get_features(epochs.get_data())
    clf, acc = create_classifier(features, labels)
    print(f'k-fold validation accuracy: {acc}')

def get_epochs(raw, trial_duration):
    events = mne.find_events(raw)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), tmin=0, tmax=trial_duration, picks="data", baseline=(0,0))
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


if __name__ == "__main__":
    main()
