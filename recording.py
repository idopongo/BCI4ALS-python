from psychopy import visual, core, event
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
from board import Board
import json
from pipeline import load_pipeline, get_epochs


def main():
    params = load_params()
    pipeline = load_pipeline('Haggai')
    raw = run_session(params)
    save_raw(raw, params)


def save_raw(raw, params):
    folder_path = create_session_folder(params["subject"])
    raw.save(os.path.join(folder_path, "raw.fif"))
    with open(os.path.join(folder_path, "params.json"), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)


def create_session_folder(subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path


def load_params():
    with open(RECORDING_PARAMS_PATH) as file:
        params = json.load(file)
    return params


def run_session(params, pipeline=None):
    """
    Run a recording session, if pipeline is passed display prediction after every epoch
    """
    # create list of random trials
    trial_markers = Marker.all() * params["trials_per_stim"]
    np.random.shuffle(trial_markers)

    # open psychopy window and display starting message
    win = visual.Window(units="norm", color=(1, 1, 1), fullscr=params["full_screen"])
    start_msg = "Hit any key to start \n press Esc at any point to exit"
    visual.TextStim(win=win, pos=(0, 0.5), text=start_msg, color=(0, 0, 0)).draw()
    win.flip()
    event.waitKeys()

    # start recording
    with Board(use_synthetic=params["use_synthetic_board"]) as board:
        for i, marker in enumerate(trial_markers):
            show_stim_for_duration(win, progress_text(win, i + 1, len(trial_markers), marker),
                                   params["get_ready_duration"])
            core.wait(params["calibration_duration"])
            board.insert_marker(marker)
            show_stim_for_duration(win, marker_image(win, marker), params["trial_duration"])

            if pipeline:
                # get epoch and display prediction
                epochs, _ = get_epochs(board.get_data(), params["trial_duration"], markers=marker)
                prediction = pipeline.predict(epochs)[-1]
                txt = classification_result_txt(win, marker, prediction)
                show_stim_for_duration(win, txt, params["display_online_result_duration"])
        win.close()
        return board.get_data()


def show_stim_for_duration(win, stim, duration):
    if 'escape' in event.getKeys():
        core.quit()
    stim.draw()
    win.flip()
    core.wait(duration)
    win.flip()


def progress_text(win, done, total, stim):
    txt = visual.TextStim(win=win, text=f'trial {done}/{total}\n get ready for {Marker(stim).name}', color=(0, 0, 0),
                          bold=True, pos=(0, 0.8))
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
                           bold=True, pos=(0, 0.8), font='arial')


def marker_image(win, marker):
    return visual.ImageStim(win=win, image=Marker(marker).image_path, units="norm", size=2, color=(1, 1, 1))


if __name__ == "__main__":
    main()
