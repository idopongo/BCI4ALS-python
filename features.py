import numpy as np
from mne_features.univariate import compute_pow_freq_bands


class FeatureExtractor:
    def __init__(self):
        self.n_overlap = 250
        self.n_fft = 62
        self.n_per_seg = 18
        self.freq_bands = [8, 10, 12, 20, 30]
        self.sfreq = 125

    def set_params(self, n_fft, n_per_seg, n_overlap, freq_bands, sfreq):
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.freq_bands = freq_bands
        self.sfreq = sfreq

    def fit(self, data, labels):
        return self

    def transform(self, data):
        psd_params = {
            "welch_n_fft": self.n_fft,
            "welch_n_per_seg": self.n_per_seg,
            "welch_n_overlap": self.n_overlap,
        }
        band_power = np.array(
            [compute_pow_freq_bands(self.sfreq, epoch, self.freq_bands, psd_params=psd_params) for epoch in data])
        return band_power
