import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler


def assign_treatment(row, player_id):
    """
    helper function for treatment assignment --> treated is when your opponent is lefty
    """
    if row['winner_id'] == player_id:
        opponent_hand = row['loser_hand']
    else:
        opponent_hand = row['winner_hand']

    if opponent_hand == 'L':
        return 1
    else:
        return 0


def Match_per_player(player_data, player_id):
    """
    calculate ITE per player per year
    :param player_data: dataframe containing all player matches
    :return: ITE for player i
    """
    player_hand = player_data[player_data['winner_id'] == player_id]['winner_hand'].values[0]
    player_data['T'] = player_data.apply(lambda row: assign_treatment(row, player_id), axis=1)

    # player id is the winner
    win = player_data[player_data['winner_id'] == player_id]








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
        dist_np = 1 - data[features].T.corr()  # corr to distance
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