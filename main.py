from brainflow import BrainFlowInputParams, BoardShim, BoardIds
import time
from psychopy import visual, core
from psychopy.visual import ShapeStim, ImageStim
import psychopy.event

def main():
    win = visual.Window(units="norm")
    show_img(win, "right", 1)
    show_img(win, "pause", 1)
    show_img(win, "left", 1)
    win.close()

def show_img(win, img_name, seconds):
    img_dict = {"right": ImageStim(win=win, image="./images/right.png", units="norm", size=2),
                "left": ImageStim(win=win, image="./images/left.png", units="norm", size=2),
                "pause": ImageStim(win=win, image="./images/pause.png", units="norm", size=2), }

    img_dict[img_name].draw()
    win.update()
    core.wait(seconds)


def create_board(id: int):
    params = BrainFlowInputParams()
    board = BoardShim(id, params)
    board.prepare_session()
    return board


def start_session(n_trials=5):
    board = create_board(-1)
    board.start_stream()
    for i in range(n_trials):
        time.sleep(1)
    board.stop_stream()
    data = board.get_board_data()
    board.release_session()

if __name__ == "__main__":
    main()