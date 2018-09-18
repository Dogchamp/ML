from math_helper import calc_euclidean_distance
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# @ARG training_data: dataframe holding 'training data'
#      test:          test image's pixel data
#      k:             number of neighbors to vote

def load_data(filepath, dtype, split):
    # Load the data from the files
    print("LOADING DATA...")
    data = pd.read_csv(filepath, header=0, dtype=dtype, na_filter=False)
    print("DATA LOADED.")

    # Split the data
    return train_test_split(data, test_size=split)


def knn(test_df, training_df, k):
    for i in range(test_df.shape[0]):
        origin = test_df.iloc[i]
        distances = training_df.apply(calc_euclidean_distance(neighbor, origin), axis=1)
        distance_frame = pd.DataFrame(data={"distances": distances}, index=distances.index)
        distance_frame.sort_values(by="distances", inplace=True)
        print(distance_frame)
        k_closest = distance_frame.iloc[:k]
        print(k_closest)
