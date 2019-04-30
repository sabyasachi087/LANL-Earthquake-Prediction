# Compare Algorithms

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import mean_absolute_error
from sklearn.model_selection import train_test_split

import accoustic_sampler as acs
import data_formatter as dtFrm
import numpy as np
from model_holder import ModelHolder, load_model

model_name = 'random_forest_regressor.model';

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()
# columns_to_drop=columns_to_drop
formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=70)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()

# print(data_df.corr()['time_to_failure'])
# import sys
# sys.exit()

# data_df = data_df.drop(['acc_max','acc_min','chg_acc_max','chg_acc_min'],axis=1)
# Splitting data into test_1 and train
# train_set, test_set = train_test_split(data_df, test_size=0.01, random_state=np.random.randint(1, 1000))
# Removing all unused variable for memory management

# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure','segment_id'], axis=1)


randForReg = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=100,
           max_features='log2', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
           oob_score=False, random_state=None, verbose=2, warm_start=False)
randForReg.fit(x_train, y_train)

mh = ModelHolder(randForReg, most_dependent_columns)
mh.save(model_name)
randForReg = None
mh_new = load_model(model_name)
randForReg, most_dependent_columns = mh_new.get()

y_pred = randForReg.predict(x_train)
mas = mean_absolute_error(y_train, y_pred)
print('Mean Absolute Error', mas)
