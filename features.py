import numpy as np
from constants import *
from mne_features.univariate import compute_pow_freq_bands
from scipy import stats

FREQ_BANDS = [8, 12, 30]


def get_features(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    features = stats.zscore(band_power)
    return np.nan_to_num(features)
