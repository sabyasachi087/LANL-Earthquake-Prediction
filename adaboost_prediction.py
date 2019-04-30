from evaluate_trendline import TrendLine
import pandas as pd
from sklearn.ensemble.weight_boosting import AdaBoostRegressor

tl = TrendLine(data_type='train')
data_df = tl.get()

# y_train = data_df['time_to_failure']
# x_train_seg = data_df['segment_id']
# x_train = data_df.drop(['time_to_failure', 'segment_id'], axis=1)

tl = TrendLine(data_type='test')
data_df = tl.get()

x_test_seg = data_df['segment_id']
x_test = data_df.drop([ 'segment_id'], axis=1)

# adbReg = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)
# 
# adbReg.fit(x_train, y_train)
# 
# y_pred = adbReg.predict(x_test)

y_pred = x_test.mean(axis=1)

data_dict = {'seg_id': x_test_seg.values, 'time_to_failure': y_pred}
out_df = pd.DataFrame(data_dict)
print('Total predicted segments ', out_df.shape[0])
 
out_df.to_csv('trendline_mean_results.csv', index=False)
