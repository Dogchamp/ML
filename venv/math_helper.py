import math
import numpy as np

def calc_euclidean_distance(vals_1, vals_2):
    if len(vals_1) != len(vals_2):
        return 0

    res = 0
    for p in range(len(vals_1)):
        res += math.pow(vals_1[p] - vals_2[p], 2)
    res = math.sqrt(res)

    return res

def standardize_around_mean(df):
    return (df - df.mean()) / df.std()
