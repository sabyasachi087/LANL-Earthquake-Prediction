from evaluate_trendline import TrendLine
import pandas as pd
import numpy as np
from sklearn.metrics.regression import mean_absolute_error
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.model_selection._split import train_test_split

tl = TrendLine(data_type='train')
data_df = tl.get()

train_set, test_set = train_test_split(data_df, test_size=0.2, random_state=np.random.randint(1, 1000))

y_train = train_set['time_to_failure']
x_train_seg = train_set['segment_id']
x_train = train_set.drop(['time_to_failure', 'segment_id'], axis=1)

y_test = test_set['time_to_failure']
x_test_seg = test_set['segment_id']
x_test = test_set.drop(['time_to_failure', 'segment_id'], axis=1)

adbReg = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)

adbReg.fit(x_train, y_train)

y_pred = adbReg.predict(x_test)

# y_pred = x_train.mean(axis=1)

print('MAE Score for acerage ', mean_absolute_error(y_test, y_pred))
