from constants import NOTCH_FREQ, NOTCH_WIDTH, LOW_PASS, HIGH_PASS
from mne.preprocessing import ICA

def preprocess(raw):
    raw.notch_filter(NOTCH_FREQ, notch_widths=NOTCH_WIDTH)
    raw.filter(LOW_PASS, HIGH_PASS)
    # ica = ICA(n_components=13)
    # ica.fit(raw)
    # ica.plot_components()
    return raw