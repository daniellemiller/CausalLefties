import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
pd.options.mode.chained_assignment = None


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


def player_score(df, feature='games_score', treatment_func=assign_treatment_diff):
    """
    create a dataframe with a score per player per year for ITE calculation
    :param df: all matches data
    :param feature: score feature name
    :param treatment_func: treatment function to separate case-control populations
    """
    col_names = ['player_id', 'year'] + [feature]
    filtered_df = df[['winner_id', 'loser_id', 'winner_hand', 'loser_hand', 'year',
                      f'{feature}_winner', f'{feature}_loser']].dropna()

    scores = dict()
    for player_id in tqdm(set(filtered_df.winner_id) | set(filtered_df.loser_id)):
        player_data = filtered_df[(filtered_df['winner_id'] == player_id) | (filtered_df['loser_id'] == player_id)]
        player_data['T'] = player_data.apply(lambda row: treatment_func(row, player_id), axis=1)

        # split cases due to different score assignment
        score_by_year_winner = \
            player_data[player_data['winner_id'] == player_id].rename(columns={f'{feature}_winner': feature}).groupby(
                ['T', 'year'])[feature].agg(['sum', 'count']).reset_index()

        score_by_year_loser = \
            player_data[player_data['loser_id'] == player_id].rename(columns={f'{feature}_loser': feature}).groupby(
                ['T', 'year'])[feature].agg(['sum', 'count']).reset_index()

        player_score_by_year = pd.concat([score_by_year_winner, score_by_year_loser])

        score_by_year = player_score_by_year.groupby(['year', 'T']).sum().reset_index()
        score_by_year['score'] = score_by_year['sum'] / score_by_year['count']
        scores[player_id] = score_by_year

    res = []
    for player_id in scores.keys():
        for idx, row in scores[player_id].iterrows():
            res.append([player_id, int(row['T']), row['year'], row['score']])
    return pd.DataFrame(res, columns=['player_id', 'T', 'year', 'score'])


def naive_player_score(df, treatment_func=assign_treatment_diff):
    """
    create a dataframe with wins percentage as score per player per year for ITE calculation
    :param df: all matches data
    :param treatment_func: treatment function to separate case-control populations
    :return: dataframe with score per player per year per T
    """
    scores = dict()
    for player_id in tqdm(set(df.winner_id) | set(df.loser_id)):
        # filter data to contain only players matches
        player_data = df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)]
        player_data['T'] = player_data.apply(lambda row: treatment_func(row, player_id), axis=1)
        # calculate the number win\lose matches for each player by T by year
        win_by_year = player_data[player_data.winner_id == player_id].groupby(['T', 'year']).size()
        lose_by_year = player_data[player_data.loser_id == player_id].groupby(['T', 'year']).size()
        win_lose = pd.DataFrame({'win': win_by_year, 'lose': lose_by_year}).fillna(0)
        scores[player_id] = win_lose['win']/(win_lose['win']+win_lose['lose'])

    res = []
    for player_id in scores.keys():
        for (t, year), score in scores[player_id].iteritems():
            res.append([player_id, int(t), year, score])
    return pd.DataFrame(res,columns=['player_id', 'T', 'year', 'score'])


def ITE(player_data, scores, player_id, t, mapper, feature='score', min_games=5):
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
        try:
            player_score = player_scores[(player_scores['year'] == year) & (player_scores['T'] == t)][feature].values[0]
        except:
            player_score = np.NaN
        # print(year, player_score, opponent_score, year_matches.shape, t)
        if np.isnan(player_score) or np.isnan(opponent_score):
            nans += 1
            continue
        if t == 1:
            ite[year] = (player_score - opponent_score)
        else:
            ite[year] = -(player_score - opponent_score)
    return ite, nans, not_enough_games


def Match(data, scores, mapper, feature='score', treatment_func=assign_treatment_L, random_mode=False,
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


def parse_match(match_result):
    """
    parse the matching results
    :param match_result: dictionary created by the Match function containing the ITE per player per T.
    """
    lst = []
    for player_id in match_result:
        t0_d = match_result[player_id][0]   # extract ite per player for control
        t1_d = match_result[player_id][1]    # extract ite per player for case
        for year in t0_d.keys() | t1_d.keys():
            lst.append((player_id, year, t0_d.get(year, np.NaN), t1_d.get(year, np.NaN)))

    res_df = pd.DataFrame(lst, columns=['player_id', 'year', 'ite_t0','ite_t1']).dropna()
    return res_df


if __name__ == "__main__":
    males_case = False
    if males_case:
        data = pd.read_csv(r'../../data/full_data.csv')
        out_dir = r'../../outputs/'
        mapper_path = r'../../outputs/mapper.pickle'
    else:
        data = pd.read_csv(r'../../women_data/full_data.csv')
        out_dir = r'../../women_outputs/'
        mapper_path = r'../../women_outputs/mapper.pickle'
    # workaround for weird edge-case when -1 is some reason stays
    for col in ['games_score_winner', 'games_score_loser', 'pts_score_winner', 'pts_score_loser']:
        data[col] = data[col].apply(lambda x: np.NaN if x == -1 else x)

    print("Pre-calculate treatment groups")
    if not os.path.exists(mapper_path):
        mapper = {}
        mapper['assign_treatment_L'] = calc_T_per_game(data, assign_treatment_L)
        mapper['assign_treatment_diff'] = calc_T_per_game(data, assign_treatment_diff)
        with open(mapper_path, 'wb') as f:
            pickle.dump(mapper, f)
    else:
        with open(mapper_path, 'rb') as f:
            mapper = pickle.load(f)
    print("Starting Matching by treatment and feature score")
    for f in [assign_treatment_L, assign_treatment_diff]:
        for score_type in ['naive', 'games_score', 'pts_score']:
            print('Treatment assigned: ', f.__name__, f" for {score_type}")
            if score_type == 'naive':
                scores = naive_player_score(data, treatment_func=f)
            else:
                scores = player_score(data, feature=score_type, treatment_func=f)
            matched, _ = Match(data, scores, mapper=mapper, treatment_func=f)
            parsed_matched = parse_match(matched)
            parsed_matched.to_csv(os.path.join(out_dir, f'MATCH_{f.__name__}_{score_type}.csv'), index=False)
