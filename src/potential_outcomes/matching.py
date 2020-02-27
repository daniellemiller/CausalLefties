import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def assign_treatment_L(row, player_id):
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


def assign_treatment_diff(row, player_id=1):
    """
    helper function for treatment assignment --> treated is when your opponent is lefty
    """
    return row['loser_hand'] != row['winner_hand']


def calc_T_per_game(data, treatment_func):
    res_d = dict()
    for idx, row in tqdm(data.iterrows()):
        p1,p2 = row[['winner_id','loser_id']].values.flatten()
        if p1 in res_d and p2 in res_d[p1]:
            continue
        t1 = treatment_func(row,p1)
        t2 = treatment_func(row,p2)

        res_d.setdefault(p1,dict())[p2] = (t1,t2)
        res_d.setdefault(p2,dict())[p1] = (t2,t1)
    return res_d

def player_score(df, feature='games_score'):
    """
    create a dataframe with a score per player per year for ITE calculation
    :param df: all matches data
    :param feature: score feature name
    """
    col_names = ['player_id', 'year', 'player_name'] + [feature]
    win_player_df =  df[['winner_id', 'year', 'winner_name', f'{feature}_winner']].dropna()
    lose_player_df = df[['loser_id', 'year', 'loser_name', f'{feature}_loser']].dropna()

    # adapt columns names for concatenation
    win_player_df.rename(columns={win_player_df.columns[i]:col_names[i] for i in range(len(col_names))}, inplace=True)
    lose_player_df.rename(columns={lose_player_df.columns[i]: col_names[i] for i in range(len(col_names))}, inplace=True)

    all_players = pd.concat([win_player_df, lose_player_df])
    grouped_players = all_players.groupby(['player_id', 'year', 'player_name']).\
        agg({feature:['mean', 'median', 'count']}).reset_index()
    grouped_players.columns = [' '.join(col).strip() for col in grouped_players.columns.values]

    return grouped_players


def ITE(player_data, scores, player_id, t, mapper, feature='games_score mean', min_games=5):
    """
    calculate ITE by player scores per year.
    :param player_id: the id of the player
    :param scores: s dataframe with scores per player per year
    :param t: treatmet. binary.
    :param player_data: dataframe containing all the matches of the plkayer
    :param feature: the score feature (pts\ games etc..)
    :return: ITE
    """

    player_data = player_data[player_data['T'] == t]
    player_scores = scores[scores['player_id'] == player_id]
    ite = dict()
    nans = 0
    not_enough_games = 0
    for year in player_scores['year']:
        year_matches = player_data[player_data['year'] == year]
        if year_matches.shape[0] < min_games:
            not_enough_games += 1
            continue
        opponents_ids = set(list(year_matches['winner_id']) + list(year_matches['loser_id'])) - {player_id}
        # need to take the right score according to T. separate into two cases
        opponent_2_t = dict()
        opponent_2_t[0] = list()
        opponent_2_t[1] = list()
        for opponet in opponents_ids:
            opponent_2_t[mapper[player_id][opponet][1]].append(opponet)

        opponent_score_t0 = scores[(scores['player_id'].isin(opponent_2_t[0])) &
                                   (scores['year'] == year) & (scores['T'] == 0)][feature].values
        opponent_score_t1 = scores[(scores['player_id'].isin(opponent_2_t[1])) &
                                   (scores['year'] == year) & (scores['T'] == 1)][feature].values
        opponent_score = np.concatenate((opponent_score_t0, opponent_score_t1), axis=None).mean()

        player_score = player_scores[(player_scores['year'] == year) & (player_scores['T'] == t)][feature].values[0]
        # print(year, player_score, opponent_score, year_matches.shape, t)
        if np.isnan(player_score) or np.isnan(opponent_score):
            nans += 1
            continue
        if t == 1:
            ite[year] = (player_score - opponent_score)
        else:
            ite[year] = -(player_score - opponent_score)
    #     print(len(ite))
    return ite, nans, not_enough_games


def Match(data, scores, mapper, feature='games_score mean', treatment_func=assign_treatment_L, random_mode=False,
          min_games=5):
    """
    Matching algorithm implementation
    """

    # define mapper by treatment func
    map_treatment = mapper[treatment_func.__name__]
    d = dict()
    no_games_d = dict()
    for player_id in tqdm(scores['player_id'].unique()):
        # filter data to contain only players matches
        player_data = data[(data['winner_id'] == player_id) | (data['loser_id'] == player_id)]
        player_data['T'] = player_data.apply(lambda row: treatment_func(row, player_id), axis=1)

        if random_mode:
            player_data['T'] = player_data['T'].sample(frac=1).values

        ite_t1, _, no_games1 = ITE(player_data, scores, player_id, t=1, mapper=map_treatment, feature=feature,
                                   min_games=min_games)
        ite_t0, _, no_games0 = ITE(player_data, scores, player_id, t=0, mapper=map_treatment, feature=feature,
                                   min_games=min_games)

        d[player_id] = (ite_t0, ite_t1)
        no_games_d[player_id] = (no_games1, no_games0)
    return d, no_games_d


if __name__ == "__main__":
    data = pd.read_csv(r'../../data/full_data.csv')
    # workaround for weird edge-case when -1 is some reason stays
    for col in ['games_score_winner', 'games_score_loser', 'pts_score_winner', 'pts_score_loser']:
        data[col] = data[col].apply(lambda x: np.NaN if x == -1 else x)
    scores = player_score(data)

    t_mapping = {}
    t_mapping['assign_treatment_L'] = calc_T_per_game(data, assign_treatment_L)
    t_mapping['assign_treatment_diff'] = calc_T_per_game(data, assign_treatment_diff)

    for f in [assign_treatment_L, assign_treatment_diff]:
        print(f.__name__)
        res = Match(data, scores, treatment_func=f)
        res = res.dropna()
        ATE = np.mean(res[['ITE_T=1', 'ITE_T=0']].values.flatten())
        print(ATE)