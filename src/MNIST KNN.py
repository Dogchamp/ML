from math_helper import calc_euclidean_distance
from knn import *
import os, operator, datetime, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

num_total_columns = 783
num_pixel_columns = 782

# Define filepaths
cwd = os.getcwd()
cwd_path = os.path.dirname(cwd)
data_filepath = cwd_path + os.sep + 'data' + os.sep + 'mnist digits' + os.sep + 'train.csv'

# Load and parse data
data = pd.read_csv(data_filepath, header=0, dtype=np.int32, na_filter=False)
x = data.drop('label', axis=1).values
y = data['label'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42, stratify=y)

target_indeces = []
for i in range(y_test.shape[0]):
    if y_test[i] == 1:
        target_indeces.append(i)

y_test = y_test[target_indeces]
x_test = x_test[target_indeces]

np.set_printoptions(threshold=100)
print(x_test[0, 130:155])

# Set up structs to hold results
neighbors = np.arange(1, 9)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

start_time = datetime.datetime.utcnow()

print("KNN")
for i, k in enumerate(neighbors):
    print(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_acc[i] = knn.score(x_train, y_train)
    test_acc[i] = knn.score(x_train, y_train)

end_time = datetime.datetime.utcnow()

run_time = end_time - start_time
print(run_time)
print(train_acc)
print(test_acc)

sns.set(style='darkgrid')
plt.plot(train_acc)
plt.savefig('train_accuracy')
plt.plot(test_acc)
plt.savefig('test_accuracy.png')
plt.show()
#training_data, test_data = load_data(filepath=training_data_file, dtype=np.int32, split=.3)



"""
# Iterate through the test rows
for row in range(test_data.shape[0]):
    test_instance = test_data.iloc[row]
    d = np.empty(shape=(0,2))

    # Go thru each row in the training data and compare its distance to the test instance
    for i in range(training_data.shape[0]):
        if i % 5000 == 0:
            print(i)
        training_instance = training_data.iloc[i]
        to_append = np.array([[i, calc_euclidean_distance(test_instance[1:], training_instance[1:])]]) # The first column of the training data is the label so we want to skip that
        d = np.append(d, to_append, axis=0)

    d = d[d[:,1].argsort()]
    k_nearest = d[:k]
    print("K NEAREST")
    print(k_nearest)

    labels = []

    for mapping in k_nearest:
        index = mapping[0]
        print(training_data.iloc[int(index)])
        labels.append(training_data.iloc[int(index)][0])

    votes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for label in labels:
        votes[label] += 1
    votes_sorted = sorted(votes.items(), key=operator.itemgetter(1))

    print("VOTES SORTED")
    print(votes_sorted)

    print("\nPREDICTION====================================================================")
    print(votes_sorted[-1][0])
    print("\nACTUAL========================================================================")
    print(test_instance[0])
"""
