from recording import record_data, load_rec_params
from pipeline import create_and_fit_pipeline, evaluate_pipeline, load_recordings, get_epochs, \
    grid_search_pipeline_hyperparams, save_hyperparams, load_hyperparams


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


def find_best_hyperparams_for_subject(subject):
    raw, rec_params = load_recordings(subject)
    epochs, labels = get_epochs(raw, rec_params['trial_duration'])
    best_hyperparams = grid_search_pipeline_hyperparams(epochs, labels)
    save_hyperparams(best_hyperparams, subject)


if __name__ == "__main__":
    # find_best_hyperparams_for_subject("Haggai2")
    create_pipeline_for_subject("Ido2")
