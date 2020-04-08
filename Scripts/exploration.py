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
PLOT_DIR = os.path.join(OUTPUT_DIR, 'Plots')

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

def data_plots(train_df, test_df):
    ## Plot the count of each target variable in the training set
    sns.countplot(x='target', data=train_df)

    plt.title('Target class count')

    plt.savefig(os.path.join(PLOT_DIR, 'target_cnt.png'))
    plt.close()

    ## Plot the distribution of word count and length
    ### Create length and word count features in both dataframes
    train_df['word_count'] = train_df['text'].apply(lambda text: len(text.split(' ')))
    train_df['length'] = train_df['text'].apply(lambda text: len(text))
    test_df['word_count'] = test_df['text'].apply(lambda text: len(text.split(' ')))
    test_df['length'] = test_df['text'].apply(lambda text: len(text))
    ### First plot the distribution for length and word count with train and test on the same axes
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    ### Word count distributions
    sns.distplot(train_df['word_count'], ax=ax1, label='Training Set')
    sns.distplot(test_df['word_count'], ax=ax1, label='Testing Set')

    ### Length distributions
    sns.distplot(train_df['length'], ax=ax2, label='Training Set')
    sns.distplot(test_df['length'], ax=ax2, label='Testing Set')

    ax1.legend()
    ax2.legend()
    plt.suptitle('Training and test set distributions')

    plt.savefig(os.path.join(PLOT_DIR, 'word_cnt_length_dists.png'))
    plt.close()

    ## Plot the word count and length distributions of the training set for each target variable
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    sns.distplot(train_df[train_df['target'] == 1]['word_count'], ax=ax1, label='Disaster')
    sns.distplot(train_df[train_df['target'] == 0]['word_count'], ax=ax1, label='No Disaster')

    sns.distplot(train_df[train_df['target'] == 1]['length'], ax=ax2, label='Disaster')
    sns.distplot(train_df[train_df['target'] == 0]['length'], ax=ax2, label='No Disaster')

    ax1.legend()
    ax2.legend()
    plt.suptitle('Target class distributions')

    plt.savefig(os.path.join(PLOT_DIR, 'target_class_dists.png'))
    plt.close()


def main():
    training = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    testing = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    print(training.columns)
    print(training.head())
    class_balance(training, 'target')

    # training = lowercase_df(training)
    # tokenization(training)
    # lowercase_df(training)
    data_plots(training, testing)


if __name__ == "__main__":
    main()
