import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class BaselineModel:
    """
    A simple baseline model that predicts the most frequent class.
    """
    def __init__(self):
        self.most_frequent_class = None

    def fit(self, X, y):
        self.most_frequent_class = y.value_counts().idxmax()

    def predict(self, X):
        return np.full(X.shape[0], self.most_frequent_class)

    def evaluate(self, X, y, pos_label='t'):  
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, pos_label=pos_label)
        }
