# Compare Algorithms

from sklearn.linear_model.base import LinearRegression
from sklearn.metrics.regression import mean_absolute_error
from sklearn.model_selection import train_test_split

import accoustic_sampler as acs
import data_formatter as dtFrm
import numpy as np
from model_holder import ModelHolder, load_model

model_name = 'linear_regressor.model';

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()
# columns_to_drop=columns_to_drop
formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=50)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()
# print(data_df.head(5))
# sys.exit()

# data_df = data_df.drop(['acc_max','acc_min','chg_acc_max','chg_acc_min'],axis=1)
# Splitting data into test_random_forest and train
# train_set, test_set = train_test_split(data_df, test_size=0.01, random_state=np.random.randint(1, 1000))
# Removing all unused variable for memory management

# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure','segment_id'], axis=1)

# y_test = test_set['time_to_failure']
# x_test_seg = test_set['segment_id']
# x_test = test_set.drop(['time_to_failure'], axis=1)
# x_test = x_test.drop(['segment_id'], axis=1)

model = LinearRegression(n_jobs=4) 
model.fit(x_train, y_train)

mh = ModelHolder(model, most_dependent_columns)
mh.save(model_name)
model = None
mh_new = load_model(model_name)
model, most_dependent_columns = mh_new.get()

print('Evaluating test data , transforming test data now ... ')
print('Calculating score and error .. ')
y_pred = model.predict(x_train)
print('Score', model.score(x_train, y_train))

mas = mean_absolute_error(y_train, y_pred)
print('Mean Absolute Error', mas)
