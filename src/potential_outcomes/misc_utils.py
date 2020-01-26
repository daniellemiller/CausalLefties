import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = 10, 8


def plot_propensity_diff(data):
    """
    density plot of propensity scores separated by treatment
    :param data: input data frame containing a 'Propensity' column
    :return: propensity score plot
    """
    sns.kdeplot(data[data['T'] == 1]['Propensity'], label='T=1')
    sns.kdeplot(data[data['T'] == 0]['Propensity'], label='T=0')
    plt.show()


def factorize_non_numeric_values(data):
    """
    factorization of categorial features
    :param data: input dataframe
    :return: data frame with numeric columns
    """
    cols_2_consider = data.select_dtypes(np.object).columns.to_list()
    for c in cols_2_consider:
        data[c] = pd.factorize(data[c])[0]
    return data
