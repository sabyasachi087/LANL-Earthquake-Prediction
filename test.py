# Compare Algorithms
import sys

from matplotlib import pyplot as plt
import scipy
from sklearn.linear_model.base import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

import accoustic_sampler as acs
import data_formatter as dtFrm
import numpy as np
import numpy as np
import pickle as pickle

degree = 2
columns_to_keep = 50

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()
# columns_to_drop=columns_to_drop
formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=50)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()
# data_df['abs_max_roll_mean_10_log'] = np.log(data_df['abs_max_roll_mean_10'].values)
# data_df['min_roll_std_1000_auc'] = np.log(np.multiply(data_df['auc'].values, data_df['min_roll_std_1000'].values))

# cols_to_log_trainsform = ['q99', 'q01', 'q01_roll_mean_10', 'q99_roll_mean_10', 'q99_roll_std_10', 'std_roll_std_10', \
#                           'std_roll_mean_10', 'std_roll_std_100', 'std_roll_std_1000', 'q99_roll_std_100', 'abs_max_roll_std_10', \
#                           'max_roll_std_10', 'abs_max_roll_std_100', 'max_roll_std_100', 'min_roll_mean_10']
# for col in cols_to_log_trainsform:
#     data_df[col + '_log'] = data_df[col].apply(lambda x: np.log(x) if x > 0  else -1 * np.log(-1 * x))

print(data_df.corr()['time_to_failure'])
print(most_dependent_columns)
