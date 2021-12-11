import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import offline_training


def main():
    subj = input("Enter Subject Name: ")
    params = {
        'trial_duration': 2,
        'trials_per_stim': 3,
        'trial_gap': 1,
    }
    raw = offline_training.run_session(**params)
    epochs = get_epochs(raw, params['trial_duration'])
    save_raw_and_epochs(subj, raw, epochs)


def get_epochs(raw, trial_duration):
    events = mne.find_events(raw, EVENT_CHAN_NAME)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), 0, trial_duration, picks="data", baseline=(0, 0))
    return epochs


def save_raw_and_epochs(subj, raw, epochs):
    folder_path = create_session_folder(subj)
    raw.save(os.path.join(folder_path, "raw.fif"))
    epochs.save(os.path.join(folder_path, "-epo.fif"))


def create_session_folder(subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path


if __name__ == "__main__":
    main()
