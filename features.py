import numpy as np
BANDS = [(1, 4), (4, 8), (8, 12), (12, 30)]

def get_features(data):
    return np.zeros((data.shape[0], 1))