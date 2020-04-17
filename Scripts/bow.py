import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

import spacy
from spacy.util import minibatch

from sklearn.metrics import f1_score, classification_report, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

import string
import nltk

import xgboost as xgb

import random

from preprocessing import text_process

# from model_evaluation import model_cm, report, ROC_plot, score_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')

def load_data(split=0.9, random_state=None):
    """
    Splits the data into training and validation sets.
    """
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")

    # Shuffle data. Based on the choice of random state.
    train_data = data.sample(frac=1, random_state=random_state)

    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}
              for y in train_data['target'].values]
    split = int(len(train_data) * split)

    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]

    return texts[:split], train_labels, texts[split:], val_labels

def train(model, train_data, optimizer):
    """
    Trains the bow model using small batches of the training set iteratively.

    Keyword Arguments:
    model - Model to train
    train_data - Training Data (as df)
    """
    losses = {}
    # random.seed(1)
    random.shuffle(train_data)

    batches = minibatch(train_data, size=8)
    for batch in batches:
        # train_data is a list of tuples [(text0, label0), (text1, label1), ...]
        # Split batch into texts and labels
        texts, labels = zip(*batch)

        # Update model with texts and labels
        model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses

def predict(model, texts):
    # Use the model's tokenizer to tokenize each input text
    docs = [model.tokenizer(text) for text in texts]

    # Use textcat to get the scores for each doc
    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    print(textcat)
    # From the scores, find the class with the highest score/probability
    predicted_class = scores.argmax(axis=1)

    return predicted_class

def evaluate(model, texts, labels):
    """ Returns the accuracy of a TextCategorizer model.

        Arguments
        ---------
        model: ScaPy model with a TextCategorizer
        texts: Text samples, from load_data function
        labels: True labels, from load_data function

    """
    # Get predictions from textcat model (using your predict method)
    predicted_class = predict(model, texts)

    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    # A boolean or int array indicating correct predictions
    correct_predictions = predicted_class == true_class

    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()

    # Find f1 score
    score = score_model(true_class, predicted_class)

    return accuracy, score

def save_model(model, model_name):
    """
    Output the trained model to disk in the Output/Models Directory
    """
    model.to_disk(os.path.join(MODEL_DIR, model_name))

def spacy_model():
    ## Text classification with spacy and a bag of words model

    ## Load the data
    train_texts, train_labels, val_texts, val_labels = load_data()
    # print(train_texts)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')

    # Create an empty model
    nlp = spacy.blank("en")

    # Create the TextCategorizer with exclusive classes and "bow" architecture
    textcat = nlp.create_pipe(
                  "textcat",
                  config={
                    "exclusive_classes": True,
                    "architecture": "bow"})

    # Add the TextCategorizer to the empty model
    nlp.add_pipe(textcat)

    # Add labels to text classifier
    textcat.add_label("NEGATIVE")
    textcat.add_label("POSITIVE")

    # spacy.util.fix_random_seed(1)
    # random.seed(1)

    optimizer = nlp.begin_training()
    train_data = list(zip(train_texts, train_labels))
    losses = train(nlp, train_data, optimizer)
    print(losses['textcat'])

    accuracy, _ = evaluate(nlp, val_texts, val_labels)
    print(f"Accuracy: {accuracy:.4f}")

    n_iters = 3
    for i in range(n_iters):
        losses = train(nlp, train_data, optimizer)
        accuracy, score = evaluate(nlp, val_texts, val_labels)
        print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f} \t f1_score: {score:.3f}")

    save_model(nlp, 'Basic_BOW')

    # predictions = predict(nlp, val_texts).tolist()
    # true_labels = [int(each['cats']['POSITIVE']) for each in val_labels]
    # print(type(predictions))
    # print(type(true_labels))
    # print(model_cm(true_labels, predictions, True))
    # report(true_labels, predictions)
    # ROC_plot(true_labels, predictions)

def sklearn_pipeline():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')

    tweet_19 = train_df.loc[19]['text']
    print('Original Tweet: ', tweet_19)
    tweet_19_cleaned = text_process(tweet_19)
    print('Processed Tweet: ', tweet_19_cleaned)

    X = train_df['text']
    y = train_df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.2)

    classifier_dict = {'MNB':MultinomialNB(),
                       'LSVC':LinearSVC(dual=False),
                       'poly_SVM':SVC(kernel='poly', C=100),
                       'sig_SVM':SVC(kernel='sigmoid', C=1.00),
                       "RBF_SVM":SVC(gamma=0.0001, C=1000000.0)}

    pipeline = Pipeline(steps = [('bow_transformer', CountVectorizer(analyzer=text_process, ngram_range=(1,2))),
                                # ('tfidf_transformer', TfidfTransformer()),
                                ('mnb', MultinomialNB())
                                ])

    scoring = {'f1': 'f1_macro', 'Accuracy': make_scorer(accuracy_score)}
    CV = cross_validate(pipeline, X, y, cv=5, scoring=scoring, verbose=1, n_jobs=1)
    # print(CV)
    print("Cross-Validation Accuracy: {0:.3} \n Cross-Validation F1 Score: {1:.3}".format(CV['test_Accuracy'].mean(), CV['test_f1'].mean()))

    # for key, value in classifier_dict.items():
    #     print("Classifier " + key)
    #     pipeline = Pipeline(steps = [('bow_transformer', CountVectorizer(analyzer=text_process)),
    #                                 # ('tfidf_transformer', TfidfTransformer()),
    #                                 (key, value)
    #                                 ])
    #     CV = cross_validate(pipeline, X, y, cv=5, scoring=scoring, verbose=1, n_jobs=1)
    #     # print(CV)
    #     print("Cross-Validation Accuracy: {0:.3} \n \
    #     Cross-Validation F1 Score: {1:.3}".format(CV['test_Accuracy'].mean(), CV['test_f1'].mean()))

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    print(classification_report(y_val, preds))

def main():
    # sklearn_pipeline()
    spacy_model()

if __name__ == "__main__":
    main()
    # score_model()
