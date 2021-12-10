from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual, core
from psychopy.visual import ImageStim
import numpy as np
from brainflow.data_filter import DataFilter
import os
from datetime import datetime
import pickle
from pathlib import Path
from enum import IntEnum

BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"


class Marker(IntEnum):
    right = 1
    left = 2
    idle = 3
    stop = 4

    @property
    def image_path(self):
        return os.path.join(IMAGES_DIR, f'{self.name}.png')


STIMULI_MARKERS = [Marker.right, Marker.left, Marker.idle]


def main():
    subj = input("Enter Subject Name: ")
    recording = run_session()
    samples, labels = split_into_samples(recording)
    save_session_data(subj, recording, samples)


def save_session_data(subj, recording, samples):
    # Create folder for session
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)

    # Save Recording
    recording_filename = f'{folder_name}_recording.csv'
    DataFilter.write_file(recording, os.path.join(folder_path, recording_filename), 'w')

    # Save Samples
    samples_filename = f'{folder_name}_samples.p'
    pickle.dump(samples, open(os.path.join(folder_path, samples_filename), "wb"))


def show_stimulus(win, stim):
    ImageStim(win=win, image=Marker(stim).image_path, units="norm", size=2).draw()
    win.update()


def create_board():
    params = BrainFlowInputParams()
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    return board


def run_session(num_per_stim=1, stim_duration=1, stim_gap=1):
    stimuli = np.tile(STIMULI_MARKERS, num_per_stim)
    np.random.shuffle(stimuli)

    # start recording
    board = create_board()
    board.start_stream()

    # display trials
    win = visual.Window(units="norm")
    for stim in stimuli:
        show_stimulus(win, stim)
        board.insert_marker(stim)
        core.wait(stim_duration)
        win.flip()  # hide stimulus
        board.insert_marker(Marker.stop)
        core.wait(stim_gap)

    # stop recording
    recording = board.get_board_data()
    board.stop_stream()
    board.release_session()

    return recording


def split_into_samples(recording):
    marker_channel = BoardShim.get_marker_channel(BOARD_ID)
    signal_channels = BoardShim.get_eeg_channels(BOARD_ID)

    signal = recording[signal_channels]
    markers = recording[marker_channel]

    # The indexes of the nonzero markers indicate stimuli start/end
    marker_idxs = np.nonzero(markers)[0]

    # Even markers mark the beginning of a stimuli, odd markers mark the end of a stimuli
    even_markers = marker_idxs[::2]
    odd_markers = marker_idxs[1::2]
    labels = markers[even_markers]
    samples = np.vstack([signal[:, pair[0]:pair[1]] for pair in zip(even_markers, odd_markers)])

    return samples, labels


if __name__ == "__main__":
    main()
