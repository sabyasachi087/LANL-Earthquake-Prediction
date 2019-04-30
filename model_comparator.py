# Compare Algorithms
import accoustic_sampler as acs
import data_formatter as dtFrm
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.linear_model.base import LinearRegression
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_absolute_error
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble.forest import RandomForestRegressor
import sys
# load dataset

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()
data_df = sampler.get()

formatter = dtFrm.LANLDataFormatter(data_df=data_df, data_type='train', doTransform=True)
data_df = formatter.transform()

# data_df = data_df.drop(['acc_max', 'acc_min', 'chg_acc_max', 'chg_acc_min'], axis=1)

# Splitting data into test_random_forest and train
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data_df, test_size=0.4, random_state=42)
# Separate output from inputs
y_train = data_df['time_to_failure']
x_train_seg = data_df['segment_id']
x_train = data_df.drop(['time_to_failure'], axis=1)
x_train = x_train.drop(['segment_id'], axis=1)

y_test = test_set['time_to_failure']
x_test_seg = test_set['segment_id']
x_test = test_set.drop(['time_to_failure'], axis=1)
x_test = x_test.drop(['segment_id'], axis=1)

# prepare models
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
svReg = SVR(C=20.299419990722537, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
  gamma=0.06841395086207253, kernel='rbf', max_iter=-1, shrinking=True,
  tol=0.001, verbose=True);

randForReg = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=100,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=800, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

models.append(('LassoReg', Lasso(alpha=0.1)))
models.append(('SVM', svReg))
models.append(('LinearReg', LinearRegression()))
models.append(('randForest', randForReg))

mas = make_scorer(mean_absolute_error, greater_is_better=False);
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, n_jobs=4)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Classification Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.grid()
pyplot.show()
