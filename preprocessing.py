import mne
import numpy as np

from board import EEG_CHAN_NAMES


class Preprocessor:
    def __init__(self):
        self.epoch_tmin = 1
        self.l_freq = 7
        self.h_freq = 30

    def set_params(self, epoch_tmin, l_freq, h_freq):
        self.epoch_tmin = epoch_tmin
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        epochs = self.laplacian(epochs)
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq, verbose=False)
        epochs = epochs[:, :, int(125 * self.epoch_tmin):]
        return epochs

    def laplacian(self, epochs):
        LAPLACIAN = {
            "C3": ["FC5", "FC1", "CP5", "CP1"],
            "Cz": ["FC1", "FC2", "CP1", "CP2"],
            "C4": ["FC2", "FC6", "CP2", "CP6"]
        }
        LAPLACIAN = {EEG_CHAN_NAMES.index(key): [EEG_CHAN_NAMES.index(chan) for chan in value] for key, value in
                     LAPLACIAN.items()}
        filtered_epochs = np.copy(epochs)
        for chan, adjacent_chans in LAPLACIAN.items():
            print(chan)
            print(adjacent_chans)
            try:
                filtered_epochs[:, chan, :] -= np.mean(epochs[:, adjacent_chans, :], axis=1)
            except:
                print()
        return filtered_epochs


def preprocess(raw):
    raw.load_data()
    raw.filter(7, 30)
    return raw


def reject_epochs(epochs, labels):
    rejected_max_val = 200 * 1e-6
    rejected_min_val = 5 * 1e-6

    bad_epochs = dict()
    for epoch_idx, epoch in enumerate(epochs):
        reasons = {
            'bad_chans': [],
            'reason': []
        }
        for i in range(len(epoch[:, 1])):
            curr_chan = epoch[i, :]
            if abs(curr_chan.min()) < rejected_min_val:
                reasons['bad_chans'].append(i)
                reasons['reason'].append('too low')
            elif abs(curr_chan.max()) > rejected_max_val:
                reasons['bad_chans'].append(i)
                reasons['reason'].append('too high')

        if len(reasons['bad_chans']) > 2:
            bad_epochs[epoch_idx] = reasons

    print(f"{len(bad_epochs.keys())} epochs removed")
    print(bad_epochs)
    return np.delete(epochs, list(bad_epochs.keys()), axis=0), np.delete(labels, list(bad_epochs.keys()), axis=0)


def find_average_voltage(epochs):
    vol_per_chan = {}
    for chan_inx in range(len(epochs[0, :, 0])):
        vol_per_chan[chan_inx + 1] = np.mean(epochs[:, chan_inx, :])
    return vol_per_chan
