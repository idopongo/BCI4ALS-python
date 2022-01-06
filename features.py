import numpy as np
from constants import *
from mne_features.univariate import compute_pow_freq_bands, compute_hjorth_complexity_spect, compute_hjorth_complexity, \
    compute_katz_fd, compute_spect_entropy, compute_rms, compute_hjorth_mobility_spect, compute_higuchi_fd, \
    compute_wavelet_coef_energy

FREQ_BANDS = [8, 12, 30]


def get_features(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS, ratios="all") for epoch in data])
    f1 = np.array([compute_hjorth_complexity_spect(FS, epoch) for epoch in data])
    f2 = np.array([compute_katz_fd(epoch) for epoch in data])
    f3 = np.array([compute_spect_entropy(FS, epoch) for epoch in data])
    f4 = np.array([compute_rms(epoch) for epoch in data])
    f5 = np.array([compute_hjorth_mobility_spect(FS, epoch) for epoch in data])
    f6 = np.array([compute_higuchi_fd(epoch) for epoch in data])
    f7 = np.array([compute_wavelet_coef_energy(epoch) for epoch in data])

    features = np.hstack((band_power, f1, f2, f3, f4, f5, f6, f7))
    return features
