
from pipeline import load_pipeline, load_recordings, get_epochs
from sklearn.metrics import confusion_matrix
import numpy as np


def main():
    model_subject_name = 'Haggai'
    data_subject_name = 'Haggai2'

    # model
    pipeline = load_pipeline(model_subject_name)

    # data
    raw, params = load_recordings(data_subject_name)
    epochs, labels = get_epochs(raw, params["trial_duration"])
    epochs = epochs.get_data()

    # evaluate
    predictions = pipeline.predict(epochs)

    # statistics
    conf_matrix = get_confusion_matrix(labels, predictions)
    print('confusion matrix (row=label, column=prediction):')
    print(conf_matrix)

    rates = calculate_true_and_false_rates(conf_matrix)
    print('true positive and false positive rates (row=label, column=true or false):')
    print(rates)


def calculate_true_and_false_rates(conf_matrix):
    num_classes = 3
    rates = np.zeros((num_classes, 2))

    # true positive
    for i in range(0, 3):
        true_label_list = conf_matrix[i]
        total_num_true = sum(true_label_list)
        hits = true_label_list[i]
        rates[i][0] = hits/total_num_true

    # false positive
    for i in range(0, 3):
        num_false_pos = 0
        num_true_pos = 0
        for j in range(0, 3):
            num_pred = conf_matrix[i, j]
            if i == j:
                num_true_pos = num_pred
            else:
                num_false_pos = num_false_pos + num_pred
        rates[i][1] = num_false_pos/(num_false_pos + num_true_pos)

    return rates


def get_confusion_matrix(labels,predictions):
    conf = confusion_matrix(labels, predictions)
    return conf


if __name__ == "__main__":
    main()


