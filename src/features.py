import numpy as np
from mne_features.univariate import compute_pow_freq_bands


class FeatureExtractor:
    def __init__(self):
        self.n_overlap = 250
        self.n_fft = 62
        self.n_per_seg = 18
        self.freq_bands = [8, 12, 30]

    def set_params(self, n_fft, n_per_seg):
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_per_seg // 2

    def fit(self, data, labels):
        return self

    def transform(self, epochs):
        psd_params = {
            "welch_n_fft": self.n_fft,
            "welch_n_per_seg": self.n_per_seg,
            "welch_n_overlap": self.n_per_seg // 2,
        }
        if psd_params["welch_n_overlap"] > psd_params["welch_n_fft"]:
            psd_params["welch_n_overlap"] = psd_params["welch_n_fft"] - 1

        sfreq = 125
        band_power = np.array(
            [compute_pow_freq_bands(sfreq, epoch, self.freq_bands, psd_params=psd_params) for epoch in epochs]
        )
        return band_power
