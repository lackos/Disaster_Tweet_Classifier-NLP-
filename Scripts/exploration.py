import pandas as pd
import numpy as np

import os

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


def main():
    training = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id")
    testing = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col = 'id')
    print(training.columns)
    print(training.head())
    # class_balance(training, 'target')

if __name__ == "__main__":
    main()
