import numpy as np
from sklearn.metrics import roc_curve
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def compute_EER(embeddings, labels, dataset_name):

    positive_pairs = []
    negative_pairs = []

    for idx1, idx2 in combinations(range(len(labels)), 2):
        if labels[idx1] == labels[idx2]:
            positive_pairs.append((idx1, idx2))
        else:
            negative_pairs.append((idx1, idx2))


    scores = []
    y_true = []
    test_score = []
    for idx1, idx2 in tqdm(positive_pairs + negative_pairs, desc='Calculating EER'):
        emb_CNN_1 = embeddings[idx1]
        emb_CNN_2 = embeddings[idx2]

        score = cosine_similarity([emb_CNN_1], [emb_CNN_2])[0][0]

        scores.append(score)
        y_true.append(1 if (idx1, idx2) in positive_pairs else 0)


    scores = np.array(scores)
    y_true = np.array(y_true)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    save_name = dataset_name + 'roc_curve_data' + '.csv'
    roc_data.to_csv(save_name, index=False)

    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.jpg', dpi=300)
    plt.close()
    return EER
