
from sklearn.metrics.regression import mean_absolute_error
from sklearn.model_selection import train_test_split

import accoustic_sampler as acs
import data_formatter as dtFrm
import lightgbm as lgb
import numpy as np
from model_holder import ModelHolder, load_model

model_name = 'lgbm_regressor.model'

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()
# columns_to_drop=columns_to_drop
formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True, doScale=True, cols_to_keep=70)
data_df = formatter.transform()
most_dependent_columns = formatter.getMostImpCols()

train_set, test_set = train_test_split(data_df, test_size=0.01, random_state=np.random.randint(1, 1000))
# Removing all unused variable for memory management
data_df = None;formatter = None;sampler = None
# Separate output from inputs
y_train = train_set['time_to_failure']
x_train_seg = train_set['segment_id']
x_train = train_set.drop(['time_to_failure'], axis=1)
x_train = x_train.drop(['segment_id'], axis=1)

y_test = test_set['time_to_failure']
x_test_seg = test_set['segment_id']
x_test = test_set.drop(['time_to_failure'], axis=1)
x_test = x_test.drop(['segment_id'], axis=1)

model = lgb.LGBMRegressor(bagging_fraction=0.5, bagging_freq=2, bagging_seed=5,
       boosting='gbdt', boosting_type='gbdt', class_weight=None,
       colsample_bytree=1.0, feature_fraction=0.5, importance_type='split',
       learning_rate=0.001, max_depth=-1, metric='mae',
       min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=100,
       min_split_gain=0.0, n_estimators=2000, n_jobs=-1, num_leaves=64,
       objective='regression', random_state=None, reg_alpha=1.0,
       reg_lambda=0.5, silent=True, subsample=1.0,
       subsample_for_bin=200000, subsample_freq=0, verbosity=-1)

model.fit(x_train, y_train, verbose=1000)
 
# Create an variable to pickle and open it in write mode
mh = ModelHolder(model, most_dependent_columns)
mh.save(model_name)
model = None
mh_new = load_model(model_name)
model, most_dependent_columns = mh_new.get()
  
y_pred = model.predict(x_test)
mas = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error', mas)








# params = {'num_leaves': 256,
#          'min_data_in_leaf': 50,
#          'objective': 'regression',
#          'max_depth':-1,
#          'learning_rate': 0.001,
#          "boosting": "gbdt",
#           "feature_fraction": 0.5,
#          "bagging_freq": 2,
#          "bagging_fraction": 0.5,
#          "bagging_seed": 0,
#          "metric": 'mae',
#          "verbosity":-1,
#          'reg_alpha': 0.25,
#          'reg_lambda': 1.0,
#          }
#  
# from sklearn.model_selection import RandomizedSearchCV
#   
# param_distributions = {"num_leaves": [64, 128, 256, 512], "min_data_in_leaf": [50, 100, 200, 400], 'objective': ['regression'],
#          'max_depth':[-1],
#          'learning_rate': [0.01, 0.001, 0.0001],
#          "boosting": ["gbdt"],
#           "feature_fraction": [0.5, 0.75, 1.0],
#          "bagging_freq": [2],
#          "bagging_fraction": [0.5],
#          "bagging_seed": [0, 5, 42],
#          "metric": ['mae'],
#          "verbosity":[-1],
#          'reg_alpha': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
#          'reg_lambda': [0.25, 0.5, 0.75, 1.0],
#          'n_estimators':[100, 500, 1000, 2000, 4000, 8000, 16000] }
#    
# from sklearn.metrics import  make_scorer
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
#    
# rnd_search_cv = RandomizedSearchCV(lgb.LGBMRegressor(), param_distributions, n_iter=1000, verbose=2, n_jobs=-1, cv=2, random_state=42, scoring=mae_scorer)
# rnd_search_cv.fit(x_train, y_train)
#   
# print(rnd_search_cv.best_estimator_)
# 
# sys.exit()
