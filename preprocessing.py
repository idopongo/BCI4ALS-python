import mne


class Preprocessor:
    def __init__(self):
        self.trim_epoch = 1
        self.l_freq = 7
        self.h_freq = 30

    def set_params(self, trim_epoch, l_freq, h_freq):
        self.trim_epoch = trim_epoch
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        sfreq = epochs.info['sfreq']
        epochs = epochs.get_data()
        epochs = mne.filter.filter_data(epochs, sfreq, self.l_freq, self.h_freq)
        epochs = epochs[:, :, int(sfreq * self.trim_epoch):]
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
