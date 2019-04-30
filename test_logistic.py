# Run logistic regression

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import accoustic_sampler as acs
import data_formatter as dtFrm
from model_holder import ModelHolder, load_model
import numpy as np

model_name = 'logistic_regression.model';

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()

formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=50)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()

# data_df = data_df.drop(['acc_max','acc_min','chg_acc_max','chg_acc_min'],axis=1)
# Splitting data into test_random_forest and train
# train_set, test_set = train_test_split(data_df, test_size=0.01, random_state=np.random.randint(1, 1000))

# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure'], axis=1)
x_train = x_train.drop(['segment_id'], axis=1)

# y_test = test_set['time_to_failure']
# x_test_seg = test_set['segment_id']
# x_test = test_set.drop(['time_to_failure'], axis=1)
# x_test = x_test.drop(['segment_id'], axis=1)

y_train = y_train.apply(lambda x:int(round(x)))

logReg = LogisticRegression(solver='lbfgs', random_state=np.random.randint(1, 1000), max_iter=200000, n_jobs=4, multi_class='auto', verbose=2)
logReg.fit(x_train, y_train)

# Create an variable to pickle and open it in write mode
mh = ModelHolder(logReg, most_dependent_columns)
mh.save('logistic_regression.model')
logReg = None
mh_new = load_model(model_name)
logReg, most_dependent_columns = mh_new.get()
y_pred = logReg.predict(x_train) 
# y_pred = pd.Series(y_pred).apply(lambda x: float(x / 10))

print('MAE for Logistic', mean_absolute_error(y_train, y_pred))
