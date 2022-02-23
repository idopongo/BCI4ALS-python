from recording import run_session, save_raw, load_params
import os
from pipeline import create_and_fit_pipeline, evaluate_pipeline
from figures import create_and_save_plots


def record_data():
    params = load_params()
    raw = run_session(params)
    folder_path = save_raw(raw, params)
    create_and_save_plots(os.path.basename(folder_path))
    return raw


def record_and_create_pipeline():
    params = load_params()
    raw = record_data()
    pipeline, epochs, labels = create_and_fit_pipeline(raw, params)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def record_create_pipeline_to_online():
    params = load_params()
    pipeline = record_and_create_pipeline()
    raw = run_session(params, pipeline)
    folder_path = save_raw(raw, params)
    create_and_save_plots(os.path.basename(folder_path))


if __name__ == "__main__":
    record_data()
