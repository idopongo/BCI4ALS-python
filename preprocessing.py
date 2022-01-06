from constants import NOTCH_FREQ, NOTCH_WIDTH, LOW_PASS, HIGH_PASS


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
    raw.notch_filter(NOTCH_FREQ, notch_widths=NOTCH_WIDTH)
    raw.filter(LOW_PASS, HIGH_PASS)
    return raw
