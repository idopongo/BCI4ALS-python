from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual, core

win = visual.Window()
msg = visual.TextStim(win, text=u"Hello Ido")

msg.draw()
win.flip()
core.wait(1)
win.close()


def create_board(id: int):
    params = BrainFlowInputParams()
    board = BoardShim(id, params)
    board.prepare_session()
    return board

def record(n_trials = 5):
    board = create_board(-1)
    board.start_stream()
    for i in range(n_trials):

        time.sleep(1)
    board.stop_stream()
    data = board.get_board_data()
    board.release_session()


