import numpy as np
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def auc_score_func(preds, labels):
    auc_score=[]
    for c in range(preds.shape[1]):
        preds_flat = preds[:,c].flatten()
        labels_flat =1*np.array(labels.flatten()==c)
        auc_score.append(roc_auc_score(labels_flat, preds_flat))
    return auc_score

def classification_report_func(preds, labels, names):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat, target_names=names)
