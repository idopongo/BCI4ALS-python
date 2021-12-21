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
EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)
MARKER_CHANNEL = BoardShim.get_marker_channel(BOARD_ID)
FS = BoardShim.get_sampling_rate(BOARD_ID)
EEG_CHAN_NAMES = BoardShim.get_eeg_names(BOARD_ID)