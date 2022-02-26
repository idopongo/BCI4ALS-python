from recording import record_data
from pipeline import create_and_fit_pipeline, evaluate_pipeline, get_epochs, \
    grid_search_pipeline_hyperparams

from data_utils import load_recordings, load_hyperparams, save_hyperparams


def record_and_create_pipeline(rec_params):
    raw = record_data(rec_params)
    pipeline, epochs, labels = create_and_fit_pipeline(raw, rec_params)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def record_create_pipeline_to_online(rec_params):
    pipeline = record_and_create_pipeline(rec_params)
    record_data(rec_params, pipeline)


def create_pipeline_for_subject(subject):
    raw, rec_params = load_recordings(subject)
    hyperparams = load_hyperparams(subject)
    pipeline, epochs, labels = create_and_fit_pipeline(raw, rec_params, hyperparams=hyperparams)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def find_best_hyperparams_for_subject(subject):
    raw, rec_params = load_recordings(subject)
    epochs, labels = get_epochs(raw, rec_params['trial_duration'], reject_bad=not rec_params['use_synthetic_board'])
    best_hyperparams = grid_search_pipeline_hyperparams(epochs, labels)
    save_hyperparams(best_hyperparams, subject)


if __name__ == "__main__":
    create_pipeline_for_subject("Ido5")
