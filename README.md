# Kaggle Disaster tweet competition
[https://www.kaggle.com/c/nlp-getting-started]

This objective of this competition is to use NLP methods to determine if a tweet
is about a natural disaster or not, a two class classification problem. A large
training data set of ~ 7000 tweets is supplied and a testing set for submission.

## Folders
### Scripts
Contains all the scripts for exploring the data, generating the models and classifying
the model.
#### exploration.py
Functions to explore the training and test data to make decisions on which model to implement.
#### model_evaluation.py
Functions to evaluate the results of the trained model on the validation set. Explores both single statistic summaries such as f1 measure (used by Kaggle for scoring) and AUR. Also contains functions to plot the confusion matrix and rOC curves.
#### bow.py
Basic 'Bag of words' classifier using spaCy package. Based on Kaggle tutorial.
#### tutorial.py
Tutorial code used to generate the sample_submission.csv file. Uses a vectorized words model and then a svm model to classify the tweets.
### Data
Contains the raw training and test data sets.
#### train.csv
#### test.csv

### Output
Output of the submission and models

### Notes on the Data
* From the data exploration the percentage of positive to  negative tweets is ~42% and therefore class imbalance should not be a large problem.

### Notes on Scripts
#### exploration.py
General data exploration script for the training and test data
