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

import xgboost as xgb
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')

def hashtag_pipe(doc):
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

def tokenization(df):
    nlp = spacy.load('en_core_web_lg')

    with nlp.disable_pipes():
        doc = nlp(df['text'].values[0])
    print(doc)
    tokens = [token for token in doc]
    print('Tokens')
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)

    print('\n Entities \n')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

def lowercase_df(df):
    df['text'] = df.apply(lambda x: x['text'].casefold(), axis=1)
    return df



def main():
    # training = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    # testing = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    # print(training.columns)
    # print(training.head())
    # # class_balance(training, 'target')
    #
    # # training = lowercase_df(training)
    # # tokenization(training)
    #
    # nlp = spacy.load('en_core_web_lg')
    # nlp.add_pipe(hashtag_pipe)
    #
    # doc = nlp("twitter #hashtag")
    #
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #           token.shape_, token.is_alpha, token.is_stop)

    # assert len(doc) == 2
    # assert doc[0].text == 'twitter'
    # assert doc[1].text == '#hashtag'

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(nrows=3, ncols=1)
    plt.show()



if __name__ == "__main__":
    main()
