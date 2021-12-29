from brainflow import BoardIds, BoardShim

# Prerocessing
NOTCH_FREQ = 50
NOTCH_WIDTH = 2
LOW_PASS = 0.5
HIGH_PASS = 40


BOARD_ID = BoardIds.CYTON_DAISY_BOARD
IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"
EVENT_CHAN_NAME = "Stim Markers"
EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)[:-3]
# EEG_CHANNEL_NAMES = [c for c in BoardShim.get_eeg_names(BOARD_ID) if c not in ['T8', 'P3', 'P4']]
MARKER_CHANNEL = BoardShim.get_marker_channel(BOARD_ID)
FS = BoardShim.get_sampling_rate(BOARD_ID)
EEG_CHAN_NAMES = BoardShim.get_eeg_names(BOARD_ID)[:-3]

HARDWARE_SETTINGS_MSG = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110XxQ030110XxW030110XxE030110XxR030110XxT030110XxY131000XxU131000XxI131000X"