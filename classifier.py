from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def create_classifier(features, labels):
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([('LDA', lda)])
    scores = cross_val_score(clf, features, labels, cv=5)
    lda.fit(features, labels)
    return lda, scores
