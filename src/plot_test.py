import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

start_time = datetime.datetime.utcnow()
sns.set(style='darkgrid')

mu, sigma = 0, 0.1  # Mean and std dev
a = np.random.normal(mu, sigma, 1000)
data = pd.DataFrame(data=a)
sns_plot = sns.relplot(data=data)
fig = sns_plot.fig
fig.savefig("test_fig.png")
end_time = datetime.datetime.utcnow()


run_time = end_time - start_time
print(run_time)
