# Run Multi Layer Perceptron

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import accoustic_sampler as acs
import data_formatter as dtFrm
from model_holder import ModelHolder, load_model
import numpy as np
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

model_name = 'mlp_regression.model';

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()

formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=50)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()

# data_df = data_df.drop(['acc_max','acc_min','chg_acc_max','chg_acc_min'],axis=1)
# Splitting data into test_random_forest and train
# train_set, test_set = train_test_split(data_df, test_size=0.2, random_state=np.random.randint(1, 1000))

# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure', 'segment_id'], axis=1)

y_train = np.around(y_train.values, decimals=2)

# mlpReg = MLPRegressor(verbose=True, tol=0.0001, max_iter=200000, n_iter_no_change=10000, hidden_layer_sizes=(200,))
mlpReg = MLPRegressor(verbose=True, max_iter=1000)
mlpReg.fit(x_train, y_train)

# Create an variable to pickle and open it in write mode
mh = ModelHolder(mlpReg, most_dependent_columns)
mh.save('mlp_regression.model')
mlpReg = None
mh_new = load_model(model_name)
mlpReg, most_dependent_columns = mh_new.get()
y_pred = mlpReg.predict(x_train) 
# y_pred = pd.Series(y_pred).apply(lambda x: float(x / 10))

print('MAE for Multi Layer Perceptron', mean_absolute_error(y_train, y_pred))
