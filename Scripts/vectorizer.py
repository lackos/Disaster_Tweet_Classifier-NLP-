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
import preprocessing as pp

import model_evaluation as model_eval

import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'Models')
SUB_DIR = os.path.join(OUTPUT_DIR, 'Submissions')

def save_model(model, model_name):
    """
    Saves model to the model directory (creates one if it does not already exist).
    """
    ## Create model directory with title case if one does not exist.
    if not os.path.exists(os.path.join(MODEL_DIR, model_name.title())):
        os.makedirs(os.path.join(MODEL_DIR, model_name.title()))

    ## Save the model the in this directory with filename model_name.sav.
    model_file = open(os.path.join(os.path.join(MODEL_DIR, model_name.title()), model_name + '.sav'), 'wb')
    pickle.dump(model,  model_file)
    model_file.close()
    print("Model successfully saved.")

def load_model(model_name):

    model_location = os.path.join(os.path.join(MODEL_DIR, model_name.title()), model_name + '.sav')
    print(model_location)
    model_file= open(model_location, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    print("Model successfully loaded.")

    return model

def Grid_search_CV(X_train, y_train):
    """
    Hyper parameter grid search for gamma and C in radial bais SVM.
    """
    ## Define the range of values to search through for both hyper parameters
    ## Total of 169 combinations for this run.
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    ## Cross validated the results.
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    ## Print the results.
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

def Radial_SVM_Random_search_CV(X_train, y_train, samples=10):
    """
    Hyper parameter grid search for gamma and C in radial bais SVM. Uses all the
    available processors.
    """
    ## Define the range of values to search through for both hyper parameters
    ## Total of 169 combinations for this run.
    print("Starting Hyper parameter search using random grid search.")
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    ## Cross validated the results.
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = RandomizedSearchCV(SVC(), param_grid, cv=cv, scoring='f1', n_iter=samples, n_jobs=-1, verbose=3)
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    ## Print the results.
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    return grid.best_params_

def load_word_vectors(model_name, vector_name):
    """
    Loads the words vectors for each tweet.
    """
    np_file = os.path.join(os.path.join(MODEL_DIR, model_name.title()), vector_name)
    vecs = np.load(np_file)
    print("Word vectors successfully loaded.")
    return vecs

def create_word_vectors(model, model_name, output=False, disable_pipes=True):
    """
    Creates the word vectors using the spaCy en_core_web_lg core model for each
    of the tweets in the training set. 'model_name' stores the word vectors in the
    appropriate model directory.
    """
    ## Load the data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col = 'id')

    if disable_pipes == True:
        with model.disable_pipes():
            vecs = np.array([model(tweet['text']).vector for idx, tweet in train_df.iterrows()])
    else:
        model.add_pipe(hashtag_pipe)
        vecs = np.array([model(tweet['text']).vector for idx, tweet in train_df.iterrows()])
    if output == True:
        if not os.path.exists(os.path.join(MODEL_DIR, model_name.title())):
            os.makedirs(os.path.join(MODEL_DIR, model_name.title()))
        filename = open(os.path.join(os.path.join(MODEL_DIR, model_name.title()), 'training_word_vectors.npy'), 'wb')
        np.save(filename, vecs)
        print("Word vectors successfully saved.")
    return vecs

def linear_SVC(X_train, X_val, y_train, y_val):
    # Create the LinearSVC model
    LSVC = LinearSVC(dual=False)
    # Fit the model
    LSVC.fit(X_train, y_train)
    predictions = LSVC.predict(X_val)
    print(y_val.tolist())
    print(predictions)
    print("LinearSVC score: " + str(score_model(y_val, predictions)))

    return LSVC, predictions

def xgb_model(X_train, X_val, y_train, y_val):
    # Create XGBoost model
    # xg_reg = xgb.XGBClassifier(feval=score_model)
    xgb_class = xgb.XGBClassifier()
    xg_class.fit(X_train,y_train)

    ## Predict the values of the validation set
    predictions = xg_class.predict(X_val)
    # print(predictions)
    print("XGBoost Classifier score: " + str(score_model(y_val, predictions)))

    return xgb_class, predictions

def radial_SVM(X_train, X_val, y_train, y_val, params={'gamma': 0.0001, 'C': 1000000.0}):
    ## Create radial basis function SVM model (default)
    RBF_SVM = SVC(**params)
    RBF_SVM.fit(X_train, y_train)
    predictions = RBF_SVM.predict(X_val)
    # print(y_val.tolist())
    # print(predictions)
    print("Radial basis SVM score: " + str(score_model(y_val, predictions)))
    return RBF_SVM, predictions

def polynomial_SVM(X_train, X_val, y_train, y_val):
    ## Create radial basis function SVM model (default)
    poly_SVM = SVC(kernel='poly', C=100)
    poly_SVM.fit(X_train, y_train)
    predictions = poly_SVM.predict(X_val)
    # print(y_val.tolist())
    # print(predictions)
    print("polynomial basis SVM score: " + str(score_model(y_val, predictions)))
    return poly_SVM, predictions

def sigmoid_SVM(X_train, X_val, y_train, y_val):
    ## Create radial basis function SVM model (default)
    sig_SVM = SVC(kernel='sigmoid', C=1.00)
    sig_SVM.fit(X_train, y_train)
    predictions = sig_SVM.predict(X_val)
    # print(y_val.tolist())
    # print(predictions)
    print("polynomial basis SVM score: " + str(score_model(y_val, predictions)))
    return sig_SVM, predictions

def create_submission(filename, preds_test, X_test):
    """
    Creates a submission file for the Kaggle competition.
    """
    submission = pd.DataFrame({'id': X_test.index, 'target': preds_test})
    submission.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def main(create_sub=False):
    ## Load the data
    train_df, test_df = pp.load_data()

    ## Lowercase all the text in the tweets
    train_df = pp.lowercase_df(train_df)
    test_df = pp.lowercase_df(test_df)

    nlp = spacy.load('en_core_web_lg')

    # vecs = create_word_vectors(nlp, 'linear_vectorizer_lower_case_hashtag', output=True, disable_pipes=False)
    vecs = load_word_vectors('linear_vectorizer_lower_case', 'training_word_vectors.npy')

    X_train, X_val, y_train, y_val = train_test_split(vecs, train_df['target'], test_size=0.1, random_state=1)

    ## Hyper-parameter optimization
    # Grid_search_CV(X_train, y_train)
    # best_params = Radial_SVM_Random_search_CV(X_train, y_train, 40)

    ## Fit and run models
    # The best parameters are {'gamma': 0.001, 'C': 10000.0} with a score of 0.77
    # rbf_svm, valid_preds = radial_SVM(X_train, X_val, y_train, y_val, {'gamma': 0.1, 'C': 10.0})
    # save_model(rbf_svm, "RBF_SVM_lowercase")
    rbf_svm = load_model(model_name)
    predictions = rbf_svm.predict(X_val)


    # polynomial_SVM(X_train, X_val, y_train, y_val)

    ## Save the model and generate the performance report.
    model_name = "RBF_SVM_lowercase"
    model_dir = os.path.join(MODEL_DIR, model_name.title())
    model_eval.generate_model_report(model_dir, model_name, y_val, predictions)

    if create_sub == True:
        # Apply model on Test data set for submission
        with nlp.disable_pipes():
            test_vectors = np.array([nlp(tweet['text']).vector for idx, tweet in test_df.iterrows()])
        # nlp.add_pipe(hashtag_pipe)
        # test_vectors = np.array([nlp(tweet['text']).vector for idx, tweet in train_df.iterrows()])

        preds_test = rbf_svm.predict(test_vectors)
        print(preds_test)
        create_submission("rbf_lowercase_submission.csv", preds_test, test_df)

if __name__ == "__main__":
    main()
