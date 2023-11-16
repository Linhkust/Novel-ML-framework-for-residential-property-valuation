from sklearn.metrics import r2_score
from tpot import TPOTRegressor
import warnings
import h2o
from h2o.automl import H2OAutoML
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from bohb_rfecv import bohb_rfecv_bohb
from matplotlib import pyplot as plt
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


# Using TPOT AutoML framework to derive the optimal ML configuration
def benchmark_tpot(data):
    x_train, x_test, y_train, y_test = train_test_set(data, train_size=0.8)
    tpot = TPOTRegressor(generations=5,
                         population_size=15,
                         verbosity=1,
                         random_state=0,
                         n_jobs=-1,
                         scoring='r2')
    start = time.time()
    tpot.fit(x_train, y_train)
    finish = time.time()

    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = tpot.predict(x_test)

    score = r2_score(y_test, y_pred)
    return running_time, score, tpot.fitted_pipeline_


# Using H2O AutoML framework to derive the optimal ML configuration
def benchmark_h2o(file_path):
    data = h2o.import_file(file_path)
    train, test = data.split_frame(ratios=[0.8], seed=0)

    x = data.columns[:-1]
    y = data.columns[-1]

    # Run AutoML for 10 base models
    aml = H2OAutoML(max_models=15, seed=0)

    start = time.time()
    aml.train(x=x, y=y, training_frame=train)
    finish = time.time()

    # the best model
    best = aml.get_best_model()

    running_time = ("%.2f" % ((finish - start) / 60))
    preds = aml.leader.predict(test)

    score = r2_score(test['Price'].as_data_frame(), preds.as_data_frame())
    return running_time, score, best


def benchmark_tpot_h2o(self, pipelines, tool):
    # identify the feature set
    temp = []
    for i, pipeline in enumerate(pipelines):
        if pipeline[0] != pipelines[i - 1][0]:
            temp.append(pipeline[0])
        else:
            pass

    benchmark_features = []
    for i, feature in enumerate(temp):
        benchmark_features.append(feature)

    benchmark_pipelines = []
    for i, feature in enumerate(benchmark_features):
        temp = []
        for j, pipeline in enumerate(pipelines):
            if pipeline[0] == feature:
                temp.append(pipeline)
            else:
                pass
        benchmark_pipelines.append(temp)

    # TPOT
    if tool == 'tpot':

        tpot_result = []
        for i, feature in enumerate(benchmark_features):
            # tpot fit
            print('TPOT Fitting for Feature Set {}:'.format(i+1))
            tpot_time, tpot_score, tpot_pipeline = benchmark_tpot(self.data[feature])
            tpot_result.append(['set {}'.format(i+1), tpot_time, tpot_score, tpot_pipeline])
        tpot_result = pd.DataFrame(tpot_result, columns=['Feature set', 'Time', 'R squared', 'Optimal Pipeline'])
        tpot_result.to_csv('./collected data/results/h2o_tpot/tpot/results.csv', index=False)

    # H2O AutoML
    elif tool == 'h2o':
        h2o.init()
        h2o_result = []
        for j, feature in enumerate(benchmark_features):
            print('H2O AutoML Fitting for Feature Set {}:'.format(j+1))
            data = self.data[feature]
            data.to_csv('./h2o_tpot/h2o/h2o_{}.csv'.format(j), index=False)
            h2o_time, h2o_score, h2o_pipeline = benchmark_h2o('./h2o_tpot/h2o/h2o_{}.csv'.format(j))
            h2o_result.append(['set {}'.format(j+1), h2o_time, h2o_score, h2o_pipeline])
        h2o_result = pd.DataFrame(h2o_result, columns=['Feature set', 'Time', 'R squared', 'Optimal Pipeline'])
        h2o_result.to_csv('./collected data/results/h2o_tpot/h2o/results.csv', index=False)

    elif tool == 'ours':
        ours_result = []
        for i, feature in enumerate(benchmark_features):
            # tpot fit
            print('AutoML_AVM Fitting for Feature Set {}:'.format(i + 1))

            start = time.time()
            pred_result = []
            for model in ['Random Forest', 'Extra Tree', 'XGBoost', 'LightGBM']:
                result = bohb_rfecv_bohb(self.data[feature], model, verbose=0)
                pred_result.append(result)

            pred_results = pd.DataFrame(pred_result,
                                        columns=['model_type', 'features', 'hyperparameters', 'Time',
                                                 'before_r2', 'before_mae', 'before_rmse',
                                                 'after_r2', 'after_mae', 'after_rmse',
                                                 'improvement'])

            max_before = pred_results['before_r2'].max()
            max_after = pred_results['after_r2'].max()

            if max_before <= max_after:
                best_pipeline = pred_results.loc[pred_results['after_r2'].idxmax()]
                ours_score = best_pipeline['after_r2']
                ours_pipeline = best_pipeline
            else:
                best_pipeline = pred_results.loc[pred_results['before_r2'].idxmax()]
                ours_score = best_pipeline['before_r2']
                ours_pipeline = best_pipeline

            finish = time.time()

            ours_result.append(['set {}'.format(i + 1), "%.2f" % ((finish - start) / 60), ours_score, ours_pipeline])
        ours_result = pd.DataFrame(ours_result, columns=['Feature set', 'Time', 'R squared', 'Optimal Pipeline'])
        ours_result.to_csv('./h2o_tpot/ours/results.csv', index=False)


def benchmark_performance(criteria):
    tpot_result = pd.read_csv('./collected data/results/h2o_tpot/tpot/results.csv')
    h2o_result = pd.read_csv('./collected data/results/h2o_tpot/h2o/results.csv')
    ours_result = pd.read_csv('./collected data/results/h2o_tpot/ours/results.csv')

    fig, ax = plt.subplots()

    if criteria == 'time':
        tpot_time = tpot_result['Time']
        h2o_time = h2o_result['Time']
        ours_time = ours_result['Time']

        boxplot1 = ax.boxplot(tpot_time, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
        boxplot2 = ax.boxplot(h2o_time, positions=[2], widths=0.6, patch_artist=True, showfliers=False)
        boxplot3 = ax.boxplot(ours_time, positions=[3], widths=0.6, patch_artist=True, showfliers=False)
        colors = ['lightblue', 'lightgreen', 'lightyellow']

        for boxplot, color in zip([boxplot1, boxplot2, boxplot3], colors):
            for patch in boxplot['boxes']:
                patch.set_facecolor(color)

        ax.set_xticklabels(['TPOT', 'H2O', 'Ours'])
        ax.set_ylabel('Time (Minutes)')
        plt.show()

    elif criteria == 'r2':
        tpot_r2 = tpot_result['R squared']
        h2o_r2 = h2o_result['R squared']
        ours_r2 = ours_result['R squared']

        boxplot1 = ax.boxplot(tpot_r2, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
        boxplot2 = ax.boxplot(h2o_r2, positions=[2], widths=0.6, patch_artist=True, showfliers=False)
        boxplot3 = ax.boxplot(ours_r2, positions=[3], widths=0.6, patch_artist=True, showfliers=False)

        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for boxplot, color in zip([boxplot1, boxplot2, boxplot3], colors):
            for patch in boxplot['boxes']:
                patch.set_facecolor(color)

        ax.set_xticklabels(['TPOT', 'H2O', 'Ours'])
        ax.set_ylabel('R squared')
        plt.show()


'''With and without features'''


def no_image_performance(self):
    selected_poi = ['MAL', 'SMK', 'KDG', 'PRS', 'SES', 'PAR', 'PLG', 'RGD', 'BUS', 'MIN', 'CPO', 'MTA']
    basic_variables = ['CCL',
                       'Floor',
                       'Area',
                       'x',
                       'y',
                       'Longitude',
                       'Latitude',
                       'wifi_hk',
                       'POI_density',
                       'Num_class',
                       'Num_type',
                       'Class_diversity',
                       'Type_diversity']

    selected_variables = []
    for threshold1 in self.poi_thresholds:
        variables = basic_variables + \
                    [x + '_Walk{}'.format(threshold1) for x in selected_poi] + ['Price']
        selected_variables.append(variables)

    pred_result = []
    for model in ['Random Forest', 'Extra Tree', 'XGBoost', 'LightGBM']:
        for selected_variable in selected_variables:
            result = bohb_rfecv_bohb(self.data[selected_variable], model, verbose=0)
            pred_result.append(result)

    pred_results = pd.DataFrame(pred_result,
                                columns=['model_type', 'features', 'hyperparameters', 'Time',
                                         'before_r2', 'before_mae', 'before_rmse',
                                         'after_r2', 'after_mae', 'after_rmse',
                                         'improvement'])

    pred_results.to_csv('./collected data/results/h2o_tpot/ours/no_image_results.csv', index=False)


def no_image_benchmark():
    ours_result = pd.read_csv('./collected data/results/h2o_tpot/ours/results.csv')
    no_image = pd.read_csv('./collected data/results/h2o_tpot/ours/no_image_results.csv')

    no_image['r2'] = no_image.apply(lambda x: x['before_r2'] if x['improvement'] == 0 else x['after_r2'], axis=1)
    ours_r2 = ours_result['R squared']
    no_image_r2 = no_image['r2']

    fig, ax = plt.subplots()
    boxplot1 = ax.boxplot(ours_r2, positions=[1], widths=0.6, patch_artist=True, showfliers=False)
    boxplot2 = ax.boxplot(no_image_r2, positions=[2], widths=0.6, patch_artist=True, showfliers=False)

    colors = ['blue', 'green']
    for boxplot, color in zip([boxplot1, boxplot2], colors):
        for patch in boxplot['boxes']:
            patch.set_facecolor(color)

    ax.set_xticklabels(['With image features', 'Without image features'])
    ax.set_ylabel('R squared')
    plt.show()