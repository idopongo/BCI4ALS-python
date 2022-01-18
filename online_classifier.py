
from recording import *
from preprocessing import preprocess
from features import get_features
import pickle
import mne


def main():
    params = load_params()
    classifier_filename = os.path.join(CLASSIFIERS_DIR, "classifier_Haggai_3.pickle")
    classifier = load_classifier(classifier_filename)
    raw = run_online_session(params, classifier)
    save_raw(raw, params)


def run_online_session(params, classifier):
    trial_stims = Marker.all() * params["trials_per_stim"]
    np.random.shuffle(trial_stims)
    # start recording
    board = create_board()
    board.config_board(HARDWARE_SETTINGS_MSG)
    board.start_stream()
    # TODO: Add calibration text
    sleep(5)
    # display trials
    win = visual.Window(units="norm", color=(1, 1, 1))
    counter = 1
    total = len(trial_stims)
    for stim in trial_stims:
        show_stim_progress(win, counter, total, stim)
        win.update()
        sleep(params["get_ready_duration"])
        win.flip()
        sleep(params["calibration_duration"])
        show_stimulus(win, stim)
        win.update()
        board.insert_marker(stim)
        sleep(params["trial_duration"])
        win.flip()
        # sleep is necessary to properly access the real-time data
        sleep(0.5)
        num_available_data = board.get_board_data_count()
        # get accumulation of recording, from that extract the current epoch
        acc_recording_board = board.get_current_board_data(num_available_data)
        acc_recording = convert_to_mne(acc_recording_board)
        acc_recording_processed = preprocess(acc_recording)
        events = mne.find_events(acc_recording_processed)
        acc_epochs = mne.Epochs(acc_recording_processed, events, stim, tmin=0, tmax=params["trial_duration"], picks="data", baseline=(0, 0))
        current_epoch = acc_epochs[-1]
        features = get_features(current_epoch.get_data())
        prediction = classifier.predict(features)
        show_classification_result(win, stim, prediction)
        win.update()
        sleep(params["display_online_result_duration"])
        win.flip()
        counter = counter + 1
    sleep(params["get_ready_duration"])
    # stop recording
    raw = convert_to_mne(board.get_board_data())
    board.stop_stream()
    board.release_session()
    return raw


def load_classifier(classifier_filename):
    pickle_read = open(classifier_filename, "rb")
    classifier = pickle.load(pickle_read)
    pickle_read.close()
    return classifier


def show_classification_result(win, stim, prediction):
    if stim == prediction:
        msg = 'correct prediction'
        col = (0, 1, 0)
    else:
        msg = 'incorrect prediction'
        col = (1, 0, 0)
    visual.TextStim(win=win, text=f'label: {Marker(stim).name}\nprediction: {Marker(prediction).name}\n{msg}',
                    color=col,
                    bold=True, pos=(0, 0.8), font='arial').draw()


if __name__ == "__main__":
    main()

