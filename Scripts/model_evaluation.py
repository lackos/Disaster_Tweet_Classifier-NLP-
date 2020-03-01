import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

import scipy.stats as stats

import spacy
from spacy.util import minibatch

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import xgboost as xgb

import random

def model_cm(model_name, model_directory, true_labels, predicted_labels, seaborn=False):
    """
    Outputs and plots the confusion matrix for the model
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    print('\nTrue Positives(TP) = ', cm[0,0])

    print('\nTrue Negatives(TN) = ', cm[1,1])

    print('\nFalse Positives(FP) = ', cm[0,1])

    print('\nFalse Negatives(FN) = ', cm[1,0])
    if seaborn == True:
        hm = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.savefig(os.path.join(model_directory,model_name + '_cm.png'))
    return confusion_matrix(true_labels, predicted_labels)

def ROC_plot(model_name, model_directory, true_labels, predicted_labels,):
    """
    Generates a .png file of the predictions ROC scores.

    Keyword Arguments:
    model_directory -- directory for the plot.
    true_labels -- predictors for the validation set.
    predicted_labels -- predicted values of the validation set.
    """
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)

    plt.figure(figsize=(6,4))

    plt.plot(fpr, tpr, linewidth=2)

    plt.plot([0,1], [0,1], 'k--' )

    plt.rcParams['font.size']

    plt.title('ROC curve')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.savefig(os.path.join(model_directory, model_name + '_ROC.png'))


def generate_model_report(model_name, model_directory, true_labels, predicted_labels):
    """
    Generates a .md file which contains a report of a models performance.

    Presents the f1 score, AUC score, the confusion matrix, and the ROC curve of
    the predictions.

    Keyword Arguments:
    model_directory -- directory for the report to go.
    model_name -- name of the model.
    true_labels -- predictors for the validation set.
    predicted_labels -- predicted values of the validation set.
    """
    ## Create the ROC plot
    ROC_plot(model_name, model_directory, true_labels, predicted_labels,)
    cm = model_cm(model_name, model_directory, true_labels, predicted_labels, seaborn=True)

    ## Image locations:
    ROC = model_name + '_ROC.png'
    CM = model_name + '_cm.png'
    ## Create and write the scores in the report
    report = open(os.path.join(model_directory, model_name + '_report.md'), 'w')
    report.write("## Model report and score \n")
    report.write("### Single Score statistics \n")
    report.write("f1 score: " + str(round(f1_score(true_labels, predicted_labels),2)) + "\n\n")
    report.write("AUC (area under ROC curve) score: " + str(round(roc_auc_score(true_labels, predicted_labels),2)) + '\n')
    report.write("\n\n")
    report.write("### Confusion Matrix \n")
    report.write("<img src='{0}' width='150'> \n\n".format(CM) )
    report.write(classification_report(true_labels, predicted_labels))
    report.write("### ROC Curve \n")
    report.write("<img src='{0}' width='150'> \n\n".format(ROC) )
    report.write("The dashed line represents random classification.")
    report.close()

    print("Report successfully generated.")
