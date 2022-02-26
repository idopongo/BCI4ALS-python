import numpy as np
import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
from board import Board
import json
from pipeline import get_epochs
from figures import create_and_save_plots

from psychopy import visual, core, event

BG_COLOR = "black"
STIM_COLOR = "white"
SYNTHETIC_SUBJECT_NAME = "Synthetic"


def record_data(rec_params, pipeline=None):
    raw = run_session(rec_params, pipeline)
    folder_path = save_raw(raw, rec_params)
    return raw


def save_raw(raw, rec_params):
    folder_path = create_session_folder(rec_params['subject'])
    raw.save(os.path.join(folder_path, "raw.fif"))
    with open(os.path.join(folder_path, "params.json"), 'w', encoding='utf-8') as f:
        json.dump(rec_params, f, ensure_ascii=False, indent=4)
    return os.path.basename(folder_path)


def create_session_folder(subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path


def load_rec_params():
    with open(RECORDING_PARAMS_PATH) as file:
        rec_params = json.load(file)

    if rec_params["use_synthetic_board"]:
        rec_params["subject"] = SYNTHETIC_SUBJECT_NAME

    return rec_params


def run_session(params, pipeline=None):
    """
    Run a recording session, if pipeline is passed display prediction after every epoch
    """
    # create list of random trials
    trial_markers = Marker.all() * params["trials_per_stim"]
    np.random.shuffle(trial_markers)

    # open psychopy window and display starting message
    win = visual.Window(units="norm", color=BG_COLOR, fullscr=params["full_screen"])
    msg1 = f'Hello {params["subject"]}!\n Hit any key to start, press Esc at any point to exit'
    loop_through_messages(win, [msg1])

    # start recording
    with Board(use_synthetic=params["use_synthetic_board"]) as board:
        for i, marker in enumerate(trial_markers):
            # "get ready" period
            show_stim_for_duration(win, progress_text(win, i + 1, len(trial_markers), marker),
                                   params["get_ready_duration"])
            # calibration period
            core.wait(params["calibration_duration"])

            # motor imagery period
            board.insert_marker(marker)
            show_stim_for_duration(win, marker_stim(win, marker), params["trial_duration"])

            if pipeline:
                # We need to wait a short time between the end of the trial and trying to get it's data to make sure
                # that we have recorded (trial_duration * sfreq) samples after the latest marker (otherwise the epoch
                # will be too short)
                core.wait(0.5)

                # get latest epoch and make prediction
                raw = board.get_data()
                print(len(raw))
                epochs, _ = get_epochs(raw, params["trial_duration"], markers=marker)
                prediction = pipeline.predict(epochs)[-1]

                # display prediction result
                txt = classification_result_txt(win, marker, prediction)
                show_stim_for_duration(win, txt, params["display_online_result_duration"])
        core.wait(0.5)
        win.close()
        return board.get_data()


def loop_through_messages(win, messages):
    for msg in messages:
        visual.TextStim(win=win, text=msg, color=STIM_COLOR).draw()
        win.flip()
        keys_pressed = event.waitKeys()
        if 'escape' in keys_pressed:
            core.quit()
        if 'backspace' in keys_pressed:
            break


def marker_stim(win, marker):
    shape = visual.ShapeStim(win, vertices=Marker(marker).shape, fillColor=STIM_COLOR, size=.5)
    return shape


def show_stim_for_duration(win, stim, duration):
    # Adding this code here is an easy way to make sure we check for an escape event before showing every stimulus
    if 'escape' in event.getKeys():
        core.quit()

    stim.draw()  # draw stim on back buffer
    win.flip()  # flip the front and back buffers and then clear the back buffer
    core.wait(duration)
    win.flip()  # flip back to the (now empty) back buffer


def progress_text(win, done, total, stim):
    txt = visual.TextStim(win=win, text=f'trial {done}/{total}\n get ready for {Marker(stim).name}', color=STIM_COLOR,
                          bold=True, alignHoriz='center', alignVert='center')
    txt.font = 'arial'
    return txt


def classification_result_txt(win, marker, prediction):
    if marker == prediction:
        msg = 'correct prediction'
        col = (0, 1, 0)
    else:
        msg = 'incorrect prediction'
        col = (1, 0, 0)
    return visual.TextStim(win=win, text=f'label: {Marker(marker).name}\nprediction: {Marker(prediction).name}\n{msg}',
                           color=col,
                           bold=True, alignHoriz='center', alignVert='center', font='arial', )


def marker_image(win, marker):
    return visual.ImageStim(win=win, image=Marker(marker).image_path, units="norm", size=2, color=(1, 1, 1))


if __name__ == "__main__":
    rec_params = load_rec_params()
    record_data(rec_params)
