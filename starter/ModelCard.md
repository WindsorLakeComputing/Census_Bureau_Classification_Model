Model Details

Ben Kelly created the model. It is a LGBM Classifier from lightgbm. It was chosen to use after testing 10 models from the "10 popular classification methods" hosted on https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods.

This model should be used to predict the salary of American citizens based off a handful of attributes. The users are prospective advertisors.
Metrics

The model was evaluated using beta score. The value is 0.7130136986301371.
Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The target class was modified from 14 categories down to 8: "workclass","education","marital-status","occupation","relationship","race","sex","native-country"

The original data set has 32,562 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label salary was used on the labels.
Bias

When the education category is vertically sliced there is an appearences of bias. The following values "12th" and "5th-6th" have a beta score of zero. The
holder of the next highest beta score, 0.2222222222222222, is "7th-8th". And the next higher score after that, 0.2857142857142857, is held by "9th". The fact
that the poorest predictions are for education scores with a grade school value demonstrates that something is awry with prediciting the salary of people
who only have a grade school education.