import mne


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
        epochs = mne.filter.filter_data(epochs, 125, self.l_freq, self.h_freq)
        epochs = epochs[:, :, int(125 * self.epoch_tmin):]
        return epochs


def laplacian(raw):
    def laplacian_C3(chan):
        avg = (raw["FC5"][0] + raw["FC1"][0] + raw["CP5"][0] + raw["CP1"][0]) / 4
        return chan - avg.squeeze()

    def laplacian_C4(chan):
        avg = (raw["FC2"][0] + raw["FC6"][0] + raw["CP2"][0] + raw["CP6"][0]) / 4
        return chan - avg.squeeze()

    def laplacian_Cz(chan):
        avg = (raw["FC1"][0] + raw["FC2"][0] + raw["CP1"][0] + raw["CP2"][0]) / 4
        return chan - avg.squeeze()

    raw.apply_function(laplacian_C3, picks=["C3"])
    raw.apply_function(laplacian_C4, picks=["C4"])
    raw.apply_function(laplacian_Cz, picks=["Cz"])


def preprocess(raw):
    raw.load_data()
    raw.filter(7, 30)
    return raw


def reject_epochs(epochs):
    rejected_max_val = 300 * 1e-6
    rejected_min_val = 5 * 1e-6
    epochs = epochs.get_data()  # array of shape (n_epochs, n_channels, n_times)

    bad_epoch = []
    for epoch_idx, epoch in enumerate(epochs):
        bad_chan = []
        for i in range(len(epoch[:, 1])):
            curr_chan = epoch[i, :]
            if abs(curr_chan.min()) < rejected_min_val:
                bad_chan.append(i)
            elif abs(curr_chan.max()) > rejected_max_val:
                bad_chan.append(i)

        if len(bad_chan) > 3:
            bad_epoch.append(epoch_idx)








