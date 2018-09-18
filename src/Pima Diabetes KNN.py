from math_helper import calc_euclidean_distance, standardize_around_mean
from knn import knn
import csv, os, operator
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

# Constants
cwd = os.getcwd()
cwd_path = os.path.dirname(cwd)
training_data_file = cwd_path + os.sep + 'data' + os.sep + 'pima-indians-diabetes-database' + os.sep + 'diabetes.csv'
k = 11

print("LOADING TRAINING DATA")
data = pd.read_csv(training_data_file, header=0)
print("TRAINING DATA LOADED")

# Standardize data (but not the label)
data.iloc[:, :-1] = standardize_around_mean(data.iloc[:, :-1])

# Split the data
training_data, test_data = train_test_split(data, test_size=.3)

print("\nDATA:")
with pd.option_context('display.max_rows', 10, 'display.max_columns', 10, 'display.width', 1000):
    print("TRAINING_DATA")
    print(training_data)

tp = 0
tn = 0
fp = 0
fn = 0

start_time = datetime.datetime.utcnow()
# Iterate through the test rows
for row in range(test_data.shape[0]):
    test_instance = test_data.iloc[row]
    d = np.empty(shape=(0,2))

    # Go thru each row in the training data and compare its distance to the test instance
    for i in range(training_data.shape[0]):
        training_instance = training_data.iloc[i]
        to_append = np.array([[i, calc_euclidean_distance(test_instance, training_instance)]]) # The first column of the training data is the label so we want to skip that
        d = np.append(d, to_append, axis=0)

    d = d[d[:,1].argsort()]
    k_nearest = d[:k]
    labels = []

    for mapping in k_nearest:
        index = mapping[0]
        labels.append(training_data.iloc[int(index)][-1])

    votes = {0: 0, 1: 0}
    for label in labels:
        votes[label] += 1
    votes_sorted = sorted(votes.items(), key=operator.itemgetter(1))

    prediction = str(votes_sorted[-1][0])
    actual = str(int(test_instance[-1]))

    if prediction == '1' and actual == '1':
        tp += 1
    if prediction == '0' and actual == '0':
        tn += 1
    if prediction == '0' and actual == '1':
        fn += 1
    if prediction == '1' and actual == '0':
        fp += 1

end_time = datetime.datetime.utcnow()

run_time = end_time - start_time
print(run_time)

print("TRUE POSTIIVES: " + str(tp))
print("TRUE NEGATIVES: " + str(tn))
print("FALSE NEGATIVES: " + str(fn))
print("FALSE POSITIVES: " + str(fp))
print("Accuracy:")
print((tp+tn)/(tp+tn+fp+fn)*100)

