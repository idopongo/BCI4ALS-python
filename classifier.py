from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

def create_classifier(features, labels):
    lda = LinearDiscriminantAnalysis()
    # csp = CSP(n_components=4, reg=None, norm_trace=False)
    # clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(lda, features, labels, cv=5)
    lda.fit(features, labels)
    return lda, scores