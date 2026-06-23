"""Predictive-performance metrics and bootstrap confidence intervals.

ROC / precision-recall plotting and bootstrapped 95% confidence intervals for
AUROC, average precision, sensitivity, and specificity, as reported in the
FairED evaluation.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve


def PlotROCCurve(probs,y_test_roc, ci= 95, random_seed=0):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    print("AUC:",roc_auc)
    print("AUPRC:", average_precision)
    print("Sensitivity:",sensitivity)
    print("Specificity:",specificity)
    print("Score thresold:",threshold)
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)


    plt.title('Receiver Operating Characteristic: AUC={0:0.4f}'.format(
          roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    precision, recall, threshold2 = precision_recall_curve(y_test_roc, probs)
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AUPRC={0:0.4f}'.format(
          average_precision))
    plt.show()
    return [roc_auc, average_precision, sensitivity, specificity, threshold, lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity]


def auc_with_ci(probs,y_test_roc, lower = 2.5, upper = 97.5, n_bootstraps=200, rng_seed=10):
    print(lower, upper)
    y_test_roc = np.asarray(y_test_roc)
    bootstrapped_auroc = []
    bootstrapped_ap = []
    bootstrapped_sensitivity = []
    bootstrapped_specificity = []

    rng = np.random.default_rng(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.integers(0, len(y_test_roc)-1, len(y_test_roc))
        if len(np.unique(y_test_roc[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr, tpr, threshold = metrics.roc_curve(y_test_roc[indices],probs[indices])
        auroc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(y_test_roc[indices], probs[indices])
        a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
        sensitivity = tpr[a]
        specificity = 1-fpr[a]
        bootstrapped_auroc.append(auroc)
        bootstrapped_ap.append(ap)
        bootstrapped_sensitivity.append(sensitivity)
        bootstrapped_specificity.append(specificity)

    lower_auroc,upper_auroc = np.percentile(bootstrapped_auroc, [lower, upper])
    lower_ap,upper_ap = np.percentile(bootstrapped_ap, [lower, upper])
    lower_sensitivity,upper_sensitivity = np.percentile(bootstrapped_sensitivity, [lower, upper])
    lower_specificity,upper_specificity = np.percentile(bootstrapped_specificity, [lower, upper])

    std_auroc = np.std(bootstrapped_auroc)
    std_ap = np.std(bootstrapped_ap)
    std_sensitivity = np.std(bootstrapped_sensitivity)
    std_specificity = np.std(bootstrapped_specificity)

    return lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity


def plot_confidence_interval(dataset, metric= 'auroc', ci=95, name = 'AUROC', my_file = 'AUROC_hosp.eps', my_path = 'my_path', dpi=300):
    ci_list = [dataset['lower_'+metric].values.tolist(),dataset['upper_'+metric].values.tolist()]
    std = [(dataset[metric]-dataset['std_'+metric]).values.tolist(), (dataset[metric]+dataset['std_'+metric]).values.tolist()]
    auc = dataset[metric].values.tolist()
    y = [range(len(dataset)), range(len(dataset))]

    plt.plot(ci_list,y, '-', color='gray',linewidth=1.5)
    plt.plot(std,y,'-', color='black', linewidth=2)
    plt.plot(auc,y[0],'|k', markersize=4)
    plt.xlabel(name)
    plt.yticks(range(len(dataset)),list(dataset['Model']))
    plt.savefig(os.path.join(my_path, my_file), format='eps', dpi=dpi)
    
    plt.show()
