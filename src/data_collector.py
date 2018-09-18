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

print(data)
columns = ["Glucose", "Outcome"]
x = pd.DataFrame(data, columns=columns)
x = x.loc[x["Outcome"]==1]
print(x)
x = x.drop(labels="Outcome", axis=1)
print(x)
sns.distplot(x)
plt.savefig("GlucosePIMADiabetes.png")
plt.show()
