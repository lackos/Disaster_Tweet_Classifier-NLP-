import pandas as pd
import numpy as np

import os

import spacy
from spacy.matcher import Matcher

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from preprocessing import text_process

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

def word_counts(train_df, test_df):
    ## Apply transform to the entire data set
    # train_df['proc_text'] = train_df['text'].apply(lambda text: text_process(text))

    ## CountVectorizer
    X = train_df['text']
    disaster = train_df[train_df['target'] == 1]['text']
    no_disaster = train_df[train_df['target'] == 0]['text']

    # bow = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')
    bow = CountVectorizer(analyzer=text_process)
    bow.fit(X)
    X_bow = bow.transform(X)
    disaster_bow = bow.transform(disaster)
    no_disaster_bow = bow.transform(no_disaster)

    # print(X_bow)
    # print(bow.get_feature_names())
    # print(X_bow.toarray().sum(axis=0))
    # print(train_df['proc_text'])

    total_wc = pd.Series(data=X_bow.toarray().sum(axis=0), index=bow.get_feature_names()).sort_values(ascending=False)
    disaster_wc = pd.Series(data=disaster_bow.toarray().sum(axis=0), index=bow.get_feature_names()).sort_values(ascending=False)
    no_disaster_wc = pd.Series(data=no_disaster_bow.toarray().sum(axis=0), index=bow.get_feature_names()).sort_values(ascending=False)

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    sns.countplot(x=total_wc)

    print(no_disaster_wc)
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(nrows=3, ncols=1, figsize=(12,18))

    plt.suptitle("Most Frequent Words", fontsize=25)

    sns.barplot(x=total_wc[0:30].values, y=total_wc[0:30].index, ax=ax1)
    ax1.set_title("Complete Training Set")

    sns.barplot(x=disaster_wc[0:30].values, y=disaster_wc[0:30].index, ax=ax2)
    ax2.set_title("Disaster Tweets")

    sns.barplot(x=no_disaster_wc[0:30].values, y=no_disaster_wc[0:30].index, ax=ax3)
    ax3.set_title("Non Disaster Tweets")

    # plt.show()
    plt.savefig(os.path.join(PLOT_DIR, 'Word_counts.png'))
    plt.close()

    ## Common words between the two classifications
    disaster = set(disaster_wc.index[0:100])
    common = [word for word in no_disaster_wc.index[0:100] if word in disaster]
    print(len(common))


def main():
    training = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    testing = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    print(training.columns)
    print(training.head())

    # class_balance(training, 'target')
    # training = lowercase_df(training)
    # tokenization(training)
    # lowercase_df(training)
    # data_plots(training, testing)
    word_counts(training, testing)


if __name__ == "__main__":
    main()
