from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual
from psychopy.visual import ImageStim
import numpy as np
from brainflow.data_filter import DataFilter
import os
from datetime import datetime

RIGHT = 1
LEFT = 2
IDLE = 3
STOP = 4

STIMULI = [RIGHT, LEFT, IDLE]

IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"

STIM_IMG_PATHS = {
    RIGHT: os.path.join(IMAGES_DIR, "right.png"),
    LEFT: os.path.join(IMAGES_DIR, "left.png"),
    IDLE: os.path.join(IMAGES_DIR, "idle.png"),
}


def main():
    subj = input("Enter Subject Name: ")
    data = run_session()
    save_data(data, subj)


def save_data(data, subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    filename = f'{date_str}_{subj}.csv'
    DataFilter.write_file(data, os.path.join(RECORDINGS_DIR, filename), 'w')


def show_stimulus(win, stim):
    ImageStim(win=win, image=STIM_IMG_PATHS[stim], units="norm", size=2).draw()
    win.update()


def create_board(id: int):
    params = BrainFlowInputParams()
    board = BoardShim(id, params)
    board.prepare_session()
    return board


def run_session(trials_per_stim=1, stim_duration=2, stim_gap=1):
    trials = np.tile(STIMULI, trials_per_stim)
    np.random.shuffle(trials)

    # start recording
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = create_board(board_id)
    board.start_stream()

    # display trials
    win = visual.Window(units="norm")
    for t in trials:
        show_stimulus(win, t)
        board.insert_marker(t)
        time.sleep(stim_duration)
        win.flip()
        time.sleep(stim_gap)

    # stop recording
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    return data


if __name__ == "__main__":
    main()
