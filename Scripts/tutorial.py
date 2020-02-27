## Based on notebook https://www.kaggle.com/philculliton/nlp-getting-started-tutorial

import pandas as pd
import numpy as np

import pickle
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')

def create_submission(filename, preds_test, X_test):
    """
    Creates a submission file for the Kaggle competition.
    """
    submission = pd.DataFrame({'id': X_test.index, 'target': preds_test})
    submission.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def save_model(model, model_name):
    ## Create model directory with title case if one does not exist.
    if not os.path.exists(os.path.join(MODEL_DIR, model_name.title())):
        os.makedirs(os.path.join(MODEL_DIR, model_name.title()))
    ## Save the model the in this directory with filename model_name.sav.
    model_file = open(os.path.join(os.path.join(MODEL_DIR, model_name.title()), model_name + '.sav'), 'wb')
    pickle.dump(model,  model_file)
    model_file.close()

def main():
    ## Load in the training and testing data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')

    count_vectorizer = feature_extraction.text.CountVectorizer()

    train_vectors = count_vectorizer.fit_transform(train_df["text"])
    test_vectors = count_vectorizer.transform(test_df["text"])

    clf = linear_model.RidgeClassifier()
    scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
    print(scores)

    clf.fit(train_vectors, train_df["target"])
    save_model(clf, 'tutorial')
    preds_test = clf.predict(test_vectors)
    print(preds_test)
    create_submission("sample_submission.csv", preds_test, test_df)

if __name__ == "__main__":
    main()
