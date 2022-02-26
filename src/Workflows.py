from recording import record_data
from pipeline import create_and_fit_pipeline, evaluate_pipeline, get_epochs, \
    grid_search_pipeline_hyperparams

from data_utils import load_recordings, load_hyperparams, save_hyperparams, load_rec_params


def record_and_create_pipeline(rec_params):
    raw = record_data(rec_params)
    pipeline, epochs, labels = create_and_fit_pipeline(raw, rec_params)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def record_create_pipeline_to_online(rec_params):
    pipeline = record_and_create_pipeline(rec_params)
    record_data(rec_params, pipeline)


def create_pipeline_for_subject(subject, pipeline_type="spectral"):
    """
    Create pipline of type in: ["spectral", "csp"]
    """
    print(f'Creating pipeline for subject {subject}...')
    raw, rec_params = load_recordings(subject)
    hyperparams = load_hyperparams(subject)
    pipeline, epochs, labels = create_and_fit_pipeline(raw, rec_params, hyperparams=hyperparams,
                                                       pipeline_type=pipeline_type)
    evaluate_pipeline(pipeline, epochs, labels)
    return pipeline


def find_best_hyperparams_for_subject(subject, pipeline_type="spectral"):
    raw, rec_params = load_recordings(subject)
    epochs, labels = get_epochs(raw, rec_params['trial_duration'], reject_bad=not rec_params['use_synthetic_board'])
    best_hyperparams = grid_search_pipeline_hyperparams(epochs, labels, pipeline_type)
    save_hyperparams(best_hyperparams, subject)


if __name__ == "__main__":
    rec_params = load_rec_params()
    create_pipeline_for_subject("David3", pipeline_type="csp")