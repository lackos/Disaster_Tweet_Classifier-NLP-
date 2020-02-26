import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

import scipy.stats as stats

import spacy
from spacy.util import minibatch

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

import xgboost as xgb

import random


def score_model(true_labels, predicted_labels):
    score = f1_score(true_labels, predicted_labels)
    # print("F1 score is: " + str(round(score, 3)))
    return score

def model_cm(true_labels, predicted_labels, seaborn=False):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    print('\nTrue Positives(TP) = ', cm[0,0])

    print('\nTrue Negatives(TN) = ', cm[1,1])

    print('\nFalse Positives(FP) = ', cm[0,1])

    print('\nFalse Negatives(FN) = ', cm[1,0])
    if seaborn == True:
        hm = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.show()
    return confusion_matrix(true_labels, predicted_labels)

def report(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels))

def ROC_plot(true_labels, predicted_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)

    plt.figure(figsize=(6,4))

    plt.plot(fpr, tpr, linewidth=2)

    plt.plot([0,1], [0,1], 'k--' )

    plt.rcParams['font.size'] = 12

    plt.title('ROC curve for Predicting a Pulsar Star classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.show()
