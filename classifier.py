from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline


def create_classifier(features, labels):
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20)
    lda = LinearDiscriminantAnalysis()
    clf = make_pipeline(lda)
    scores = cross_val_score(clf, features, labels, cv=skf)
    return clf, scores
