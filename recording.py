from brainflow import BrainFlowInputParams
from psychopy import core, visual
import numpy as np
from time import sleep
import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import json

def main():
    subj = input("Enter Subject Name: ")
    params = {
        'trial_duration': 1,
        'trials_per_stim': 1,
        'trial_gap': 1,
    }
    raw = run_session(**params)
    save_raw(subj, raw, params)

def save_raw(subj, raw, params):
    folder_path = create_session_folder(subj)
    raw.save(os.path.join(folder_path, "raw.fif"))
    with open(os.path.join(folder_path, "params.json"), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

def create_session_folder(subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path

def run_session(trials_per_stim=3, trial_duration=1, trial_gap=1):
    trial_stims = np.tile(Marker.all(), trials_per_stim)
    np.random.shuffle(trial_stims)

    # start recording
    board = create_board()
    board.start_stream()

    # display trials
    win = visual.Window(units="norm")
    for stim in trial_stims:
        sleep(trial_gap)
        show_stimulus(win, stim)
        board.insert_marker(stim)
        sleep(trial_duration)
        win.flip()  # hide stimulus
    sleep(trial_gap)
    # stop recording
    raw = convert_to_mne(board.get_board_data())
    board.stop_stream()
    board.release_session()

    return raw


def show_stimulus(win, stim):
    visual.ImageStim(win=win, image=Marker(stim).image_path, units="norm", size=2).draw()
    win.update()


def create_board():
    params = BrainFlowInputParams()
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    return board


def convert_to_mne(recording):
    recording[EEG_CHANNELS] = recording[EEG_CHANNELS] / 1e6  # BrainFlow returns uV, convert to V for MNE
    data = recording[EEG_CHANNELS + [MARKER_CHANNEL]]
    ch_types = ['eeg'] * len(EEG_CHANNELS) + ['stim']
    ch_names = EEG_CHAN_NAMES + [EVENT_CHAN_NAME]
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

if __name__ == "__main__":
    main()
