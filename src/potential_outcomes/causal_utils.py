import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


def get_propensity(data, features, mdl=LogisticRegression(max_iter=1000)):
    """
    caluculates propensity scores using ML models
    :param data: input dataframe
    :param mdl: classifier default=LR
    :param features: feature column names in raw datafile
    :return: Propensity score array
    """

    pscore = cross_val_predict(mdl, data[features], data['T'], cv=10, method='predict_proba')[:, 1]
    return pscore


def ipw(data):
    """
    calculates ATT based on IPW using pre-calculated propensity score
    formula is based on EQ 12 - https://onlinelibrary.wiley.com/doi/full/10.1002/bimj.201600094 
    :return: ATT
    """
    
    first = np.sum(data['T']*data['Y']) / data['T'].sum() 
    div_propensity = data['Propensity'] / (1 - data['Propensity'])
    second_numerator = np.sum((1-data['T']) * data['Y'] * div_propensity)
    second_denumerator = np.sum((1-data['T']) * div_propensity)
    
    att = first - second_numerator / second_denumerator


    return att

def ATE_IPW(data):
    weights = 1.0 / data['Propensity'] * data['T'] - 1.0 / (1.0 - data['Propensity']) * (1 - data['T'])
    return np.mean(data['Y'] * weights)


def Match(data, features, dist_metric='euclidean', caliper=0.05, delta=100, corr_mode=False, IV=None):
    """
    implementation of the matching algorithm for ATT calculation based on 1NN matching
    # based on the paper Austin 2011
    :param data: input dataframe
    :param features: feature column names in raw datafile
    :param dist_metric: distance metric default EU
    :param caliper: threshold for regulation
    :param delta: regulation coefficient
    :param corr_mode: whether to use correlation as a distance metric
    :param IV: invert covariance matrix for data
    :return: dataframe contains neighbors information - distance and index in input
    """
    # scale data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])
    scaled = pd.DataFrame(scaled, columns=features, index=data.index)

    for feat in scaled.columns:
        data[feat] = scaled[feat]

    # calculate pairwise distance
    if corr_mode:
        dist_np = 1 - data[features].T.corr()     # corr to distance
    elif dist_metric == 'mahalanobis':
        dist = DistanceMetric.get_metric('mahalanobis', VI=IV)
        dist_np = dist.pairwise(data[features])
    else:
        dist = DistanceMetric.get_metric(dist_metric)
        dist_np = dist.pairwise(data[features])

    dist_df = pd.DataFrame(dist_np, index=data.index, columns=data.index)

    # add regularization by weighting matrix defined by caliper and delta
    pr = data['Propensity']
    reg_dict = dict()
    for i in pr.index:
        reg_dict[i] = (abs(pr[i] - pr) > caliper) * delta
    reg_df = pd.DataFrame(reg_dict, index=data.index, columns=data.index)

    # combine distance with regulation
    dist_df = reg_df + dist_df

    # filter data to contain treated (rows) and controls (columns)
    treated_idx = data[data['T'] == 1].index
    control_idx = data[data['T'] == 0].index

    filtered_data = dist_df[dist_df.index.isin(treated_idx)][control_idx]

    # return 1-nearest neighbor
    filtered_data['knn'] = filtered_data.apply(lambda row: (row.nsmallest(n=1).index.to_list(),
                                                            row.nsmallest(n=1).to_list()), axis=1)
    
    filtered_data['knn_index'] = filtered_data['knn'].apply(lambda x: x[0][0])
    filtered_data['knn_dist'] = filtered_data['knn'].apply(lambda x: x[-1][0])

    return filtered_data[['knn_dist', 'knn_index']]


def get_matching_att(data, matched):
    """
    Extract ATT from Match() results
    :param data: input dataframe
    :param matched: Match() results
    :return: ATT
    """
    matched['Y1'] = matched.apply(lambda row: data.loc[row.name]['Y'], axis=1)
    matched['Y0'] = matched.apply(lambda row: data.loc[row['knn_index']]['Y'], axis=1)

    return np.mean(matched['Y1'] - matched['Y0'])


def S_learner(data, features, model=LinearRegression(), k=10):
    """
    implementation of S learner algorithm
    :param data: input dataframe
    :param features: feature column names in raw datafile
    :param model: LM model. default linear regression
    :param k: cv folds. default 10
    :return: ATT
    """
    preds = []
    X = data[features + ['T']]
    y = data['Y']

    # use cross validation for training
    cv = KFold(k)

    # Train on T=1 data only
    for train, test in cv.split(X, y):
        X_test = X.iloc[test]
        reg = model.fit(X.iloc[train], y.iloc[train])

        X_treated = X_test[X_test['T'] == 1]

        # generate data to predict y0
        X_control = X_test[X_test['T'] == 1]
        X_control['T'] = 0

        predicted_y1 = reg.predict(X_treated)
        predicted_y0 = reg.predict(X_control)
        preds.append(predicted_y1 - predicted_y0)
    return np.mean(np.hstack(preds))


def T_learner(data, features, model_treated=LinearRegression(), model_control=LinearRegression(), k=10):
    """
    implementation of S learner algorithm
    :param data: input dataframe
    :param features: feature column names in raw datafile
    :param model_treated: LM model for treated. default linear regression
    :param model_control: LM model for control. default linear regression
    :param k: cv folds. default 10
    :return: ATT
    """
    preds = []
    X = data[features + ['T']]
    y = data['Y']

    # use cross validation for training
    cv = KFold(k)

    for train, test in cv.split(X, y):
        X_train = X.iloc[train]
        y_train = y.iloc[train]

        X_test = X.iloc[test]
        print(X_train[X_train['T'] == 1].shape)
        treated_idx = X_train[X_train['T'] == 1].index
        control_idx = X_train[X_train['T'] == 0].index

        # train 2 models - one for treatment and one for control
        reg_treated = model_treated.fit(X_train[X_train['T'] == 1], y_train.loc[treated_idx])
        reg_control = model_control.fit(X_train[X_train['T'] == 0], y_train.loc[control_idx])

        X_treated = X_test[X_test['T'] == 1]    # for ATT

        # predict treated data using both models
        predicted_y1 = reg_treated.predict(X_test)  # changed from X_treated for ATE
        predicted_y0 = reg_control.predict(X_test)


        preds.append(predicted_y1 - predicted_y0)
    return np.mean(np.hstack(preds))
