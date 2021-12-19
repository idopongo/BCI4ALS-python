from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np

def create_classifier(features, labels):
    clf = LinearDiscriminantAnalysis()
    scores = cross_val_score(clf, features, labels, cv=3)
    clf.fit(features, labels)
    return clf, np.mean(scores)