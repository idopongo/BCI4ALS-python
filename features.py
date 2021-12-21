import numpy as np
from constants import *
from mne_features.univariate import compute_pow_freq_bands

FREQ_BANDS = [0.5, 4, 8, 13, 30, 40]


def get_features(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    return band_power
