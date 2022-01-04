from brainflow import BoardIds, BoardShim

# Preprocessing
NOTCH_FREQ = 50
NOTCH_WIDTH = 3
LOW_PASS = 2
HIGH_PASS = 30

# Paths
IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"

# BrainFlow, Cyton
BOARD_ID = BoardIds.CYTON_DAISY_BOARD
# last 3 electrodes are turned off
EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)[:-3]
EEG_CHAN_NAMES = BoardShim.get_eeg_names(BOARD_ID)[:-3]
MARKER_CHANNEL = BoardShim.get_marker_channel(BOARD_ID)
EVENT_CHAN_NAME = "Stim Markers"
FS = BoardShim.get_sampling_rate(BOARD_ID)
# This Message instructs the cyton dongle to configure electrodes gain as X6, and turn off last 3 electrodes
HARDWARE_SETTINGS_MSG = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110XxQ030110XxW030110XxE030110XxR030110XxT030110XxY131000XxU131000XxI131000X"