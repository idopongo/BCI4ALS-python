from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual, core
from psychopy.visual import ShapeStim, ImageStim
import psychopy.event
import numpy as np
from brainflow.data_filter import DataFilter

RIGHT = 1
LEFT = 2
IDLE = 3
STOP = 4

STIMULI = [RIGHT, LEFT, IDLE]

IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"

stim_imgs = {
    RIGHT: IMAGES_DIR + "/right.png",
    LEFT: IMAGES_DIR + "/left.png",
    IDLE: IMAGES_DIR + "/idle.png",
}

def main():
    data = run_session()
    DataFilter.write_file(data, RECORDINGS_DIR + '/test.csv', 'w')

def show_stimulus(win, stim):
    ImageStim(win=win, image=stim_imgs[stim], units="norm", size=2).draw()
    win.update()

def create_board(id: int):
    params = BrainFlowInputParams()
    board = BoardShim(id, params)
    board.prepare_session()
    return board

def run_session(trials_per_stim=1, stim_duration = 2, stim_gap=1):
    trials = np.tile(STIMULI, trials_per_stim)
    np.random.shuffle(trials)

    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = create_board(board_id)
    board.start_stream()

    win = visual.Window(units="norm")
    for t in trials:
        show_stimulus(win, t)
        board.insert_marker(t)
        time.sleep(stim_duration)
        win.flip()
        time.sleep(stim_gap)

    board.stop_stream()
    data = board.get_board_data()
    board.release_session()

    chans = BoardShim.get_eeg_channels(board_id)
    return data

if __name__ == "__main__":
    main()