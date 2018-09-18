import csv, os, operator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math_helper import standardize_around_mean

# Constants
cwd = os.getcwd()
cwd_path = os.path.dirname(cwd)
training_data_file = cwd_path + os.sep + 'data' + os.sep + 'pima-indians-diabetes-database' + os.sep + 'diabetes.csv'

print("LOADING TRAINING DATA")
data = pd.read_csv(training_data_file, header=0)
print("TRAINING DATA LOADED")

x = data["Pregnancies"].values
y = data["Outcome"].values
y[y == 1] = 2
y[y == 0] = 1
d = pd.DataFrame(data={"Pregnancies": x, "Diabetes": y})
print(d)

sns.relplot(data=d)
plt.show()
