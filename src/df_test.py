import pandas as pd
import numpy as np

df = pd.DataFrame(data={'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]})
target_indices = [1, 2, 3]
print(df)
df = df.iloc[target_indices]
print(df)
