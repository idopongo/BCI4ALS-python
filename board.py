from brainflow import BrainFlowInputParams
from serial.tools import list_ports
from brainflow import BoardIds, BoardShim
import mne

# This Message instructs the cyton dongle to configure electrodes gain as X6, and turn off last 3 electrodes
HARDWARE_SETTINGS_MSG = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110XxQ030110XxW030110XxE030110XxR030110XxT030110XxY131000XxU131000XxI131000X "

EEG_CHAN_NAMES = ["C3", "C4", "Cz", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "O1", "O2"]
EVENT_CHAN_NAME = "Stim Markers"


class Board:
    def __init__(self, use_synthetic=False):
        params = BrainFlowInputParams()
        if use_synthetic:
            self.board_id = BoardIds.SYNTHETIC_BOARD
        else:
            self.board_id = BoardIds.CYTON_BOARD
            params.serial_port = find_serial_port()
        board = BoardShim(self.board_id, params)

        self.brainflow_board = board

    def __enter__(self):
        self.brainflow_board.prepare_session()
        self.brainflow_board.config_board(HARDWARE_SETTINGS_MSG)
        self.brainflow_board.start_stream()
        return self

    def __exit__(self, *args):
        self.brainflow_board.stop_stream()
        self.brainflow_board.release_session()

    def insert_marker(self, marker):
        self.brainflow_board.insert_marker(marker)

    def get_data(self):
        recording = self.brainflow_board.get_board_data()
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)[:-3]  # last 3 electrodes are turned off
        marker_channel = BoardShim.get_marker_channel(self.board_id)
        recording[eeg_channels] = recording[eeg_channels] / 1e6  # BrainFlow returns uV, convert to V for MNE
        data = recording[eeg_channels + [marker_channel]]
        ch_types = ['eeg'] * len(eeg_channels) + ['stim']
        ch_names = EEG_CHAN_NAMES + [EVENT_CHAN_NAME]
        sfreq = BoardShim.get_sampling_rate(self.board_id)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        raw.set_montage("standard_1020")
        return raw


def find_serial_port():
    plist = list_ports.comports()
    FTDIlist = [comport for comport in plist if comport.manufacturer == 'FTDI']
    if len(FTDIlist) > 1:
        raise LookupError(
            "More than one FTDI-manufactured device is connected. Please enter serial_port manually.")
    if len(FTDIlist) < 1:
        raise LookupError("FTDI-manufactured device not found. Please check the dongle is connected")
    return FTDIlist[0].name
