from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from constants import FS


def create_classifier(features, labels):
    pca = PCA(n_components=40)
    cv = ShuffleSplit(20, test_size=0.1)
    lda = LinearDiscriminantAnalysis()
    clf = make_pipeline(StandardScaler(), pca, lda)
    scores = cross_val_score(clf, features, labels, cv=cv)
    return clf, scores


def create_csp_classifier(epochs):
    labels = epochs.events[:, -1]
    csp = CSP(n_components=6)
    lda = LinearDiscriminantAnalysis()
    clf = make_pipeline(csp, lda)
    cv = ShuffleSplit(10, test_size=0.2)
    scores = cross_val_score(clf, epochs, labels, cv=cv)
    print("CSP Classifier accuracy: " + np.mean(scores))


def csp_test(epochs):
    epochs.load_data()
    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    labels = epochs.events[:, -1]
    epochs_train = epochs.copy().crop(tmin=1)
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    #
    # # Use scikit-learn Pipeline with cross_val_score function
    # clf = Pipeline([('CSP', csp), ('LDA', lda)])
    # scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    print(np.mean(scores))
    w_length = int(FS * 0.5)  # running classifier: window length
    w_step = int(FS * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / FS + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.33, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()
