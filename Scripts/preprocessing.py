import pandas as pd
import numpy as np

import spacy

import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC, SVC
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from model_evaluation import model_cm, report, ROC_plot, score_model

import xgboost as xgb
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col = 'id')
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    return train_df, test_df

def lowercase_df(df):
    df['text'] = df.apply(lambda x: x['text'].casefold(), axis=1)
    return df

def hashtag_pipe(doc):
    """
    https://stackoverflow.com/questions/43388476/how-could-spacy-tokenize-hashtag-as-a-whole
    """
    merged_hashtag = False
    while True:
        for token_index,token in enumerate(doc):
            if token.text == '#':
                if token.head is not None:
                    start_index = token.idx
                    end_index = start_index + len(token.head.text) + 1
                    if doc.merge(start_index, end_index) is not None:
                        merged_hashtag = True
                        break
        if not merged_hashtag:
            break
        merged_hashtag = False
    return doc

def main():
    pass

if __name__ == "__main__":
    main()
