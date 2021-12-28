from constants import NOTCH_FREQ, NOTCH_WIDTH, LOW_PASS, HIGH_PASS


def preprocess(raw):
    raw.notch_filter(NOTCH_FREQ, notch_widths=NOTCH_WIDTH)
    raw.filter(LOW_PASS, HIGH_PASS)
    # raw.preprocessing.ICA()
    # https://mne.tools/stable/generated/mne.preprocessing.ICA.html?highlight=ica#mne.preprocessing.ICA
    # todo: add ICA
    return raw
