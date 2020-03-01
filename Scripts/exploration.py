import pandas as pd
import numpy as np

import os

import spacy
from spacy.matcher import Matcher


import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

def class_balance(training_data, target_col):
    positives = training_data[training_data[target_col] == 1].count()[target_col]
    negatives = training_data[training_data[target_col] == 0].count()[target_col]

    print("Total size of training set: " + str(training_data.shape[0])   )
    print("Number of Positives: " + str(positives))
    print("Number of negatives: " + str(negatives) )
    print("Positive Ratio: " + str(round(positives/training_data.shape[0]*100, 2)) + '%')

def tokenization(df):
    nlp = spacy.load('en_core_web_lg')
    matcher = Matcher(nlp.vocab)
    matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])

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
    training = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    testing = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    print(training.columns)
    print(training.head())
    class_balance(training, 'target')

    # training = lowercase_df(training)
    # tokenization(training)
    # lowercase_df(training)


if __name__ == "__main__":
    main()
