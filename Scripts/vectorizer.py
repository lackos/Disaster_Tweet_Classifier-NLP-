import pandas as pd
import numpy as np

import spacy

import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')

def load_word_vectors(model_name, vector_name):
    np_file = os.path.join(os.path.join(MODEL_DIR, model_name.title()), vector_name)
    vecs = np.load(np_file)
    print("Word vectors successfully loaded.")
    return vecs

def create_word_vectors(model_name, output=False):
    ## Load the data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col = 'id')
    # print(train_texts)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')

    nlp = spacy.load('en_core_web_lg')

    with nlp.disable_pipes():
        vecs = np.array([nlp(tweet['text']).vector for idx, tweet in train_df.iterrows()])

    if output == True:
        if not os.path.exists(os.path.join(MODEL_DIR, model_name.title())):
            os.makedirs(os.path.join(MODEL_DIR, model_name.title()))
        filename = open(os.path.join(os.path.join(MODEL_DIR, model_name.title()), 'training_word_vectors.npy'), 'wb')
        np.save(filename, vecs)
        print("Word vectors successfully saved.")
    return vecs
def main():
    ## Load the data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col = 'id')
    # print(train_texts)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')

    nlp = spacy.load('en_core_web_lg')

    vecs = create_word_vectors('linear_vectorizer', output=True)
    # vecs = load_word_vectors('linear_vectorizer', 'training_word_vectors.npy')

    X_train, X_test, y_train, y_test = train_test_split(vecs, train_df['target'], test_size=0.1, random_state=1)

    # Create the LinearSVC model
    model = LinearSVC(random_state=1, dual=False)
    # Fit the model
    model.fit(X_train, y_train)

    

    # Uncomment and run to see model accuracy
    print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')

if __name__ == "__main__":
    main()
