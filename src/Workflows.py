from recording import run_session
from pipeline import evaluate_pipeline, get_epochs, bayesian_opt
from data_utils import load_recordings, load_hyperparams, save_hyperparams, load_rec_params
import spectral
import csp


def record_and_create_pipeline(pipeline=spectral):
    rec_params = load_rec_params()
    run_session(rec_params)
    create_pipeline_for_subject(rec_params["subject"], pipeline)
    return pipeline


def create_pipeline_for_subject(subject, pipeline=spectral, choose=False):
    """
    Create pipline of type in: ["spectral", "csp"]
    """
    print(f'Creating pipeline for subject {subject}...')
    epochs, labels = load_epochs_for_subject(subject, choose)
    hyperparams = load_hyperparams(subject)

    pipe = pipeline.create_pipeline(hyperparams)
    pipe.fit(epochs, labels)

    evaluate_pipeline(pipe, epochs, labels)
    return pipe


def find_best_hyperparams_for_subject(subject=None, pipeline=spectral, choose=False):
    epochs, labels = load_epochs_for_subject(subject, choose)
    best_hyperparams = bayesian_opt(epochs, labels, pipeline)
    save_hyperparams(best_hyperparams, subject)


def record_with_live_retraining(subject, pipeline=spectral, choose=False):
    epochs, labels = load_epochs_for_subject(subject, choose=choose)
    rec_params = load_rec_params()
    run_session(rec_params, pipeline=pipeline, live_retraining=True, epochs=epochs, labels=labels)
    create_pipeline_for_subject(rec_params["subject"], pipeline=pipeline)


def load_epochs_for_subject(subject, choose=False):
    raws, rec_params = load_recordings(subject, choose)
    epochs, labels = get_epochs(raws, rec_params["trial_duration"], rec_params["calibration_duration"],
                                reject_bad=not rec_params['use_synthetic_board'])
    return epochs, labels


if __name__ == "__main__":
    # create_pipeline_for_subject("David7", pipeline=csp)
    create_pipeline_for_subject("Ori2", pipeline=csp)
