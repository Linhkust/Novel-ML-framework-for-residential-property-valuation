import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFECV
from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import warnings
import time
warnings.filterwarnings('ignore')


def train_test_set(data, train_size):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_scale = StandardScaler().fit(x.values)
    y_scale = StandardScaler().fit(y.values.reshape(-1, 1))
    x = x_scale.transform(x.values)
    y = y_scale.transform(y.values.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y.flatten(),
                                                        train_size=train_size,
                                                        random_state=0)
    return x_train, x_test, y_train, y_test


# Bayesian optimization with Hyperband
def bohb(clf, x_train, y_train):
    param_distributions = CS.ConfigurationSpace(seed=42)

    if clf == 'Random Forest':
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("n_estimators", 10, 200))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_samples_split", 2, 21))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_samples_leaf", 1, 21))

        search = HpBandSterSearchCV(estimator=RandomForestRegressor(),
                                    param_distributions=param_distributions,
                                    random_state=0,
                                    n_iter=10,
                                    verbose=0,
                                    min_budget=0.2,
                                    max_budget=1,
                                    resource_name='n_samples',
                                    n_jobs=-1,
                                    cv=3)
        search.fit(x_train, y_train)
        return search.best_params_, search.best_score_, RandomForestRegressor()

    elif clf == 'Extra Tree':
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("n_estimators", 10, 200))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_samples_split", 2, 20))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_samples_leaf", 1, 20))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter('bootstrap', choices=[True, False]))

        search = HpBandSterSearchCV(estimator=ExtraTreesRegressor(),
                                    param_distributions=param_distributions,
                                    random_state=0,
                                    n_iter=10,
                                    verbose=0,
                                    min_budget=0.1,
                                    max_budget=1,
                                    resource_name='n_samples',
                                    n_jobs=-1,
                                    cv=3)
        search.fit(x_train, y_train)
        return search.best_params_, search.best_score_, ExtraTreesRegressor()

    elif clf == 'XGBoost':
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("n_estimators", 10, 200))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_depth", 1, 10))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_child_weight", 1, 20))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter("subsample", choices=np.arange(0.1, 1.0, 0.1)))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter('learning_rate', choices=[1e-3, 1e-2, 1e-1, 0.5]))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter('objective', choices=['reg:squarederror']))

        search = HpBandSterSearchCV(estimator=XGBRegressor(),
                                    param_distributions=param_distributions,
                                    random_state=0,
                                    n_iter=10,
                                    verbose=0,
                                    min_budget=0.1,
                                    max_budget=1,
                                    resource_name='n_samples',
                                    n_jobs=-1,
                                    cv=3)
        search.fit(x_train, y_train)
        return search.best_params_, search.best_score_, XGBRegressor()

    elif clf == 'LightGBM':
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_leaves", 20, 50))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_depth", 1, 10))
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("n_estimators", 10, 200))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter('learning_rate', choices=[1e-3, 1e-2, 1e-1, 0.5]))
        param_distributions.add_hyperparameter(CSH.CategoricalHyperparameter('verbose', choices=[-1]))

        # n_iter simply controls how many configurations are evaluated.
        search = HpBandSterSearchCV(estimator=LGBMRegressor(),
                                    param_distributions=param_distributions,
                                    random_state=0,
                                    n_iter=10,
                                    verbose=0,
                                    min_budget=0.1,
                                    max_budget=1,
                                    resource_name='n_samples',
                                    n_jobs=-1,
                                    cv=3)
        search.fit(x_train, y_train)
        return search.best_params_, search.best_score_, LGBMRegressor()


# Recursive elimination with cross validation
def rfecv(clf_name, best_params_, x_train, y_train):
    if clf_name == 'Random Forest':
        clf = RandomForestRegressor(random_state=0)
        clf.set_params(**best_params_)
        selector = RFECV(clf, step=2, cv=3, verbose=0, n_jobs=-1)
        selector.fit(x_train, y_train.ravel())
        return selector.support_

    elif clf_name == 'Extra Tree':
        clf = ExtraTreesRegressor(random_state=0)
        clf.set_params(**best_params_)
        selector = RFECV(clf, step=2, cv=3, verbose=0, n_jobs=-1)
        selector.fit(x_train, y_train.ravel())
        return selector.support_

    elif clf_name == 'XGBoost':
        clf = XGBRegressor(random_state=0)
        clf.set_params(**best_params_)
        selector = RFECV(clf, step=2, cv=3, verbose=0, n_jobs=-1)
        selector.fit(x_train, y_train.ravel())
        return selector.support_

    elif clf_name == 'LightGBM':
        clf = LGBMRegressor(random_state=0)
        clf.set_params(**best_params_)
        selector = RFECV(clf, step=2, cv=3, verbose=0, n_jobs=-1)
        selector.fit(x_train, y_train.ravel())
        return selector.support_


# Hybrid method of BOHB and RFECV
def bohb_rfecv_bohb(data, clf_name, verbose):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_scale = StandardScaler().fit(x.values)
    y_scale = StandardScaler().fit(y.values.reshape(-1, 1))
    x = x_scale.transform(x.values)
    y = y_scale.transform(y.values.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y.flatten(),
                                                        train_size=0.8,
                                                        random_state=0)

    start = time.time()

    # Include all features and search the best hyperparameters
    if verbose == 1:
        print('1. BOHB searching with all features...')
    else:
        pass
    result_before = bohb(clf_name, x_train, y_train)
    best_params_before = result_before[0]
    clf = result_before[2]

    clf.set_params(**best_params_before)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Transform to previous value
    y_test = y_scale.inverse_transform(y_test.reshape(-1, 1))
    y_pred = y_scale.inverse_transform(y_pred.reshape(-1, 1))
    score_before_rs = [r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))]
    if verbose == 1:
        print('{}: test set performance with all features: '.format(clf_name), score_before_rs)
    else:
        pass

    # After feature selection
    if verbose == 1:
        print('2. RFECV searching...')
    else:
        pass

    rfecv_features = rfecv(clf_name, best_params_before, x_train, y_train)
    selected_features = [index for index, item in enumerate(rfecv_features) if item]

    if verbose == 1:
        print('{}: {} features are selected:'.format(clf_name, len(selected_features)), selected_features)
    else:
        pass

    xx = data.iloc[:, selected_features]
    yy = data.iloc[:, -1]
    xx_scale = StandardScaler().fit(xx.values)
    yy_scale = StandardScaler().fit(yy.values.reshape(-1, 1))
    xx = xx_scale.transform(xx.values)
    yy = yy_scale.transform(yy.values.reshape(-1, 1))
    xx_train, xx_test, yy_train, yy_test = train_test_split(xx,
                                                            yy.flatten(),
                                                            train_size=0.8,
                                                            random_state=0)
    if verbose == 1:
        print('3. BOHB searching with RFECV features...')
    else:
        pass
    result_after = bohb(clf_name, xx_train, yy_train)
    best_params_after = result_after[0]
    clf = result_after[2]
    clf.set_params(**best_params_after)
    clf.fit(xx_train, yy_train)

    yy_pred = clf.predict(xx_test)
    yy_pred = y_scale.inverse_transform(yy_pred.reshape(-1, 1))
    yy_test = yy_scale.inverse_transform(yy_test.reshape(-1, 1))
    score_after_rs = [r2_score(yy_test, yy_pred), mean_absolute_error(yy_test, yy_pred), np.sqrt(mean_squared_error(yy_test, yy_pred))]

    if verbose == 1:
        print('{}: test set performance after feature selection: '.format(clf_name), score_after_rs)
    else:
        pass

    '''Compare'''
    # model with all features and bohb-based hyperparameters
    # model with rfecv-based features and afterward bohb-based hyperparameters
    if score_before_rs[0] > score_after_rs[0]:
        if verbose == 1:
            print('4. Result: Initial features with bohb hyperparameters are selected!')
        else:
            pass
        finish = time.time()
        running_time = ("%.2f" % ((finish - start) / 60))
        return [clf_name, data.iloc[:, :-1].columns.values.tolist(), best_params_before, running_time] + score_before_rs + score_after_rs + ['0']
    else:
        if verbose == 1:
            print('4. Result: RFECV features with afterward bohb hyperparameters are selected!')
        else:
            pass
        finish = time.time()
        running_time = ("%.2f" % ((finish - start) / 60))
        return [clf_name, data.iloc[:, selected_features].columns.values.tolist(), best_params_after, running_time] + score_before_rs + score_after_rs + ['1']