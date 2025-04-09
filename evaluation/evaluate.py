# evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_model(true_labels, predicted_scores):
    auc = roc_auc_score(true_labels, predicted_scores)
    print("ROC AUC:", auc)
    return auc

if __name__ == '__main__':
    # Example evaluation with dummy data
    true_labels = np.array([0, 1, 0, 1])
    predicted_scores = np.array([0.2, 0.8, 0.3, 0.7])
    evaluate_model(true_labels, predicted_scores)
