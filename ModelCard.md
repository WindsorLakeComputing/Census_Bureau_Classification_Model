Model Details

Ben Kelly created the model. It is a LGBM Classifier from lightgbm. It was chosen to use after testing 10 models from the "10 popular classification methods" hosted on https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods.

Intended Use
This model should be used to predict the salary of American citizens based off a handful of attributes. The users are prospective advertisors.

Metrics

The model was evaluated using beta score. The value is 0.7130136986301371.

Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The target class was modified from 14 categories down to 8: "workclass","education","marital-status","occupation","relationship","race","sex","native-country"

The original data set has 32,562 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label salary was used on the labels.

Evaluation Data

20% of the data is used for testing
Metrics
precision is: 0.7864823348694316
recall is:    0.6518141311266709
beta is:      0.7128437173686042

Ethical Considerations

"With great power comes great responsibility" applies here. If the knowledge acquired from this dataset is used to only target wealthy at the exclusion of
the poor then this dataset is being used for harm. If, however, the data is used to understand why different economic status group behave then this data is
being used for good.

Caveats and Recommendations

The data was acquired from a random sample so it is subject to bias. In order to acquire more accurate data it would be recommended to randomly sample
people to acquire data.
