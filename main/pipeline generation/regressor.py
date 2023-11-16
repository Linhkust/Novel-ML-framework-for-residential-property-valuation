import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from bohb_rfecv import bohb_rfecv_bohb
from matplotlib import pyplot as plt
import fileinput
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import shap

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Regressor(object):
    def __init__(self, data, poi_thresholds, gsv_thresholds, rs_thresholds, cnn_types):
        self.data = data
        self.poi_thresholds = poi_thresholds
        self.gsv_thresholds = gsv_thresholds
        self.rs_thresholds = rs_thresholds
        self.cnn_types = cnn_types

    '''Generate ML pipelines'''

    # First: data with different features
    # Second: different model configurations: ExtraTree, RandomForest, XGBoost, LightGBM
    def generate_pipelines(self):
        pipelines = []
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
            for threshold2 in self.gsv_thresholds:
                for threshold3 in self.rs_thresholds:
                    for cnn_type in self.cnn_types:
                        for dimensions in [['2d1', '2d2'], ['3d1', '3d2', '3d3']]:
                            variables = basic_variables + \
                                        [x + '_Walk{}'.format(threshold1) for x in selected_poi] + \
                                        [x + '{}'.format(threshold2) for x in ['sky', 'building', 'vegetation']] + \
                                        [x + '{}'.format(threshold3) for x in ['NDVI', 'NDWI', 'NDBI']] + \
                                        [cnn_type + '_' + dimension for dimension in dimensions] + \
                                        ['Price']
                            selected_variables.append(variables)

        models = ['Extra Tree', 'Random Forest', 'XGBoost', 'LightGBM']
        for variable in selected_variables:
            for model in models:
                pipelines.append([variable, model])  # features + model types

        return pipelines

    # use the hybrid methodof BOHB and RFECV to integrate feature selection and hyperparameter tuning
    def fit(self, data, model, verbose):
        results = bohb_rfecv_bohb(data, model, verbose)
        return results

    def pipelines_fit(self, pipelines):
        results = []
        for num, pipeline in tqdm(enumerate(pipelines), total=len(pipelines)):
            print('Pipeline_{}/{} Training:'.format(num + 1, len(pipelines)))

            result = self.fit(self.data[pipeline[0]], pipeline[1], verbose=1)
            result.insert(0, 'Pipeline_{}'.format(num + 1))
            results.append(result)

        pred_results = pd.DataFrame(results,
                                    columns=['pipeline_id', 'model_type', 'features', 'hyperparameters', 'Time',
                                             'before_r2', 'before_mae', 'before_rmse',
                                             'after_r2', 'after_mae', 'after_rmse',
                                             'improvement'])

        pred_results.to_csv('./collected data/results/Pipelines.csv', index=False)

    # This fun is used for distributed computing
    def pipelines_fit_single(self, pipeline):
        result = self.fit(self.data[pipeline[0]], pipeline[1], verbose=1)
        return result

    # This fun is used for running on one PC
    def pipelines_fit_stream(self, pipelines):
        # 定义一个列表来存储每次循环的结果
        results = []
        for num, pipeline in tqdm(enumerate(pipelines), total=len(pipelines)):
            print('Pipeline_{}/{} Training:'.format(num + 1, len(pipelines)))

            result = self.fit(self.data[pipeline[0]], pipeline[1], verbose=1)
            result.insert(0, 'Pipeline_{}'.format(num + 1))
            results.append(result)

        pred_results = pd.DataFrame(results,
                                    columns=['pipeline_id', 'model_type', 'features', 'hyperparameters', 'Time',
                                             'before_r2', 'before_mae', 'before_rmse',
                                             'after_r2', 'after_mae', 'after_rmse',
                                             'improvement'])

        pred_results.to_csv('./collected data/results/Pipelines.csv', index=False)

    # Best pipeline
    def best_pipeline_(self):
        pred_results = pd.read_csv('./collected data/results/Pipelines.csv')
        max_before = pred_results['before_r2'].max()
        max_after = pred_results['after_r2'].max()

        if max_before <= max_after:
            best_pipeline = pred_results.loc[pred_results['after_r2'].idxmax()]
        else:
            best_pipeline = pred_results.loc[pred_results['before_r2'].idxmax()]

        best_pipeline = best_pipeline.to_frame().T
        print(best_pipeline)
        return best_pipeline.reset_index()

    # Export the best pipeline as python file
    def export_pipeline(self):
        best_pipeline = self.best_pipeline_()

        # generate py file for model deployment
        file_path = './collected data/results/template.py'
        with fileinput.input(files=file_path, inplace=True) as f:
            for line_number, line in enumerate(f, start=1):
                if line_number == 10:
                    line = line.replace('[]', str([best_pipeline.loc[0, 'features']]))
                elif line_number == 18:
                    line = line.replace('{}', str(best_pipeline.loc[0, 'hyperparameters']))
                elif line_number == 19:
                    if best_pipeline.loc[0, 'model_type'] == "Extra Tree":
                        line = line.replace('Regressor', 'ExtraTreesRegressor')
                    elif best_pipeline.loc[0, 'model_type'] == 'Random Forest':
                        line = line.replace('Regressor', 'RandomForestRegressor')
                    elif best_pipeline.loc[0, 'model_type'] == 'XGBoost':
                        line = line.replace('Regressor', 'XGBRegressor')
                    elif best_pipeline.loc[0, 'model_type'] == 'LightGBM':
                        line = line.replace('Regressor', 'LGBMRegressor')
                print(line, end='')

    # Global feature importance
    def global_explain(self):
        best_pipeline = self.best_pipeline_()

        selected_features = eval(best_pipeline.loc[0, 'features'].replace('"', ''))
        data = pd.read_csv('collected data/clean data/paper.csv')
        feature_values = data[selected_features]
        training_features, testing_features, training_target, testing_target = train_test_split(feature_values,
                                                                                                data['Price'],
                                                                                                test_size=0.2,
                                                                                                random_state=0)

        # Model fit
        hyperparameters = eval(best_pipeline.loc[0, 'hyperparameters'].replace('"', ''))

        if best_pipeline.loc[0, 'model_type'] == "Extra Tree":
            model = ExtraTreesRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'Random Forest':
            model = RandomForestRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'XGBoost':
            model = XGBRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'LightGBM':
            model = LGBMRegressor(**hyperparameters)

        reg = model.fit(training_features, training_target)
        feature_importance = reg.feature_importances_

        feature_importance = (feature_importance - np.min(feature_importance)) \
                             / (np.max(feature_importance) - np.min(feature_importance)) * 100

        feature_importance = pd.DataFrame(feature_importance).T

        feature_importance.columns = selected_features

        feature_importance = feature_importance.T.sort_values(by=0, axis=0, ascending=True).copy()
        feature_importance.columns = ['Relative Importance']

        print(feature_importance)

        feature_importance.plot.barh(figsize=[12, 8], legend=False)
        plt.xlabel('Relative Importance (%)')
        plt.ylabel('Features')
        plt.savefig('global_importance.png', dpi=300)

    # Local feature importance using SHAP
    def local_explain(self, index):
        best_pipeline = self.best_pipeline_()
        hyperparameters = eval(best_pipeline.loc[0, 'hyperparameters'].replace('"', ''))
        selected_features = eval(best_pipeline.loc[0, 'features'].replace('"', ''))
        data = pd.read_csv('./collected data/clean data/paper.csv')
        feature_values = data[selected_features]
        training_features, testing_features, training_target, testing_target = train_test_split(feature_values,
                                                                                                data['Price'],
                                                                                                test_size=0.2,
                                                                                                random_state=0)

        if best_pipeline.loc[0, 'model_type'] == "Extra Tree":
            model = ExtraTreesRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'Random Forest':
            model = RandomForestRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'XGBoost':
            model = XGBRegressor(**hyperparameters)
        elif best_pipeline.loc[0, 'model_type'] == 'LightGBM':
            model = LGBMRegressor(**hyperparameters)

        reg = model.fit(training_features, training_target)
        explainer = shap.Explainer(reg, feature_names=selected_features)
        values = explainer(testing_features.iloc[index, :])
        shap.plots.bar(values, max_display=12)