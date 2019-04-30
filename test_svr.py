# Compare Algorithms
from sklearn.metrics.regression import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import accoustic_sampler as acs
import data_formatter as dtFrm
from model_holder import ModelHolder, load_model
import numpy as np

model_name = 'support_vector_regression.model';

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()
# columns_to_drop=columns_to_drop
formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=50)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()

# print(data_df.corr()['time_to_failure'])
# sys.exit()

# data_df = data_df.drop(['acc_max','acc_min','chg_acc_max','chg_acc_min'],axis=1)
# Splitting data into test_random_forest and train
# train_set, test_set = train_test_split(data_df, test_size=0.01, random_state=np.random.randint(1, 1000))
# Removing all unused variable for memory management

# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure', 'segment_id'], axis=1)


svReg = SVR(C=9137.08647605824366, cache_size=200, coef0=0.0, degree=2, epsilon=0.001,
  gamma=0.586414861763494, kernel='rbf', max_iter=-1, shrinking=True,
  tol=0.001, verbose=True)

svReg.fit(x_train, y_train)
# Create an variable to pickle and open it in write mode
mh = ModelHolder(svReg, most_dependent_columns)
mh.save(model_name)
svReg = None
mh_new = load_model(model_name)
svReg, most_dependent_columns = mh_new.get()

y_pred = svReg.predict(x_train)

mas = mean_absolute_error(y_train, y_pred)
print('Mean Absolute Error', mas)
