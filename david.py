from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual, core
from psychopy.visual import ShapeStim


win = visual.Window()
RightArrowVert = [(-0.4, 0.05), (-0.4, -0.05), (-.2, -0.05), (-.2, -0.1), (0, 0), (-.2, 0.1), (-.2, 0.05)]
arrow = ShapeStim(win, vertices=RightArrowVert, fillColor='darkred', size=.5, lineColor='red')
arrow.draw()
win.update()
core.wait(10)
win.close()


def create_board(id: int):
    params = BrainFlowInputParams()
    board = BoardShim(id, params)
    board.prepare_session()
    return board


def record(n_trials=5):
    board = create_board(-1)
    board.start_stream()
    for i in range(n_trials):
        time.sleep(1)
    board.stop_stream()
    data = board.get_board_data()
    board.release_session()
