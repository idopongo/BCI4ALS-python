from constants import NOTCH_FREQ, NOTCH_WIDTH, LOW_PASS, HIGH_PASS


def preprocess(raw):
    raw.load_data()
    raw.notch_filter(NOTCH_FREQ, notch_widths=NOTCH_WIDTH)
    raw.filter(LOW_PASS, HIGH_PASS)
    return raw
