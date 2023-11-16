import numpy as np
import pandas as pd
import scipy.stats as stats
import Orange
import matplotlib.pyplot as plt

data = pd.read_csv('./collected data/results/Pipelines.csv')


# Wilcoxon signed-ranks test for P_A and P_R
def pa_pr():
    r2_test = stats.wilcoxon(data['before_r2'], data['after_r2'], alternative='less')
    rmse_test = stats.wilcoxon(data['before_rmse'], data['after_rmse'], alternative='greater')
    print(r2_test.pvalue)
    print(rmse_test.pvalue)


def sam_improvement_test():
    test_data = pd.read_csv('./collected data/results/sam_validation.csv')
    test = stats.friedmanchisquare(test_data.iloc[0, 1:4],
    test_data.iloc[1, 1:4],
    test_data.iloc[2, 1:4],
    test_data.iloc[3, 1:4],
    test_data.iloc[4, 1:4],
    test_data.iloc[5, 1:4],
    test_data.iloc[6, 1:4])
    print(test_data[['Frankfurt', 'Lindau', 'Munster']].rank(axis=0, ascending=False, method='min'))
    print(test.pvalue)


def pipeline_test():
    values = []
    for i in range(len(data)):
        values.append(list(data.iloc[i, 5:11]))

    data_array = np.array((values))
    pipeline_before_test = stats.friedmanchisquare(*data_array.T)

    print(pipeline_before_test.pvalue)
    print(pipeline_before_test.statistic)


def pipeline_rank():
    data['before_r2_rank'] = data['before_r2'].rank(ascending=False, method='max')
    data['before_rmse_rank'] = data['before_rmse'].rank(ascending=True, method='min')
    data['after_r2_rank'] = data['after_r2'].rank(ascending=False, method='max')
    data['after_rmse_rank'] = data['after_rmse'].rank(ascending=True, method='min')

    data['average_rank'] = data.apply(lambda x: x[['before_r2_rank',
                                                   'before_rmse_rank',
                                                   'after_r2_rank',
                                                   'after_rmse_rank'
                                                   ]].mean(), axis=1)

    values = []
    for i in range(len(data)):
        values.append(list(data.loc[i, ['before_r2', 'before_rmse', 'after_r2', 'after_rmse']]))

    data_array = np.array((values))
    pipeline_before_test = stats.friedmanchisquare(*data_array.T)

    print(pipeline_before_test.pvalue)
    print(pipeline_before_test.statistic)

    data.to_csv('Rank_pipelines.csv', index=False)
