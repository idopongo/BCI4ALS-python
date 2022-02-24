from recording import run_session, save_raw, load_rec_params
import os
from pipeline import create_and_fit_pipeline, evaluate_pipeline
from figures import create_and_save_plots


def record_data(rec_params):
    raw = run_session(rec_params)
    folder_path = save_raw(raw, rec_params)
    create_and_save_plots(os.path.basename(folder_path))
    return raw


def record_and_create_pipeline(rec_params):
    raw = record_data(rec_params)
    pipeline, epochs, labels = create_and_fit_pipeline(raw, rec_params)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def record_create_pipeline_to_online(rec_params):
    pipeline = record_and_create_pipeline(rec_params)
    raw = run_session(rec_params, pipeline)
    folder_path = save_raw(raw, rec_params)
    create_and_save_plots(os.path.basename(folder_path))


if __name__ == "__main__":
    rec_params = load_rec_params()
    record_data(rec_params)
