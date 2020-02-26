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


def ITE(player_data, scores, player_id, t, feature='games_score mean'):
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
    ite = 0
    nans = 0
    for year in player_scores['year']:
        year_matches= player_data[player_data['year'] == year]
        opponents_ids = set(list(year_matches['winner_id']) + list(year_matches['loser_id'])) - {player_id}
        opponent_score = scores[(scores['player_id'].isin(opponents_ids)) & (scores['year'] == year)][feature].mean()
        player_score = player_scores[player_scores['year'] == year][feature].values[0]
        # print(year, player_score, opponent_score, year_matches.shape, t)
        if np.isnan(player_score) or np.isnan(opponent_score):
            nans += 1
            continue
        ite += (player_score - opponent_score)
    try:
        ite = ite/(player_scores.shape[0]-nans)
    except ZeroDivisionError:
        return None
    if t == 1:
        return ite
    else:
        return -ite



def Match(data, scores, feature='games_score mean', treatment_func=assign_treatment_L):
    """
    Matching algorithm implementation
    """
    dfs = []
    for player_id in tqdm(scores['player_id'].unique()):
        # filter data to contain only players matches
        player_data = data[(data['winner_id'] == player_id) | (data['loser_id'] == player_id)]
        player_data['T'] = player_data.apply(lambda row: treatment_func(row, player_id), axis=1)

        ite_t1 = ITE(player_data, scores, player_id, t=1, feature=feature)
        ite_t0 = ITE(player_data, scores, player_id, t=0, feature=feature)
        cur_df = pd.DataFrame({'player_id':player_id, 'ITE_T=0':ite_t0, 'ITE_T=1':ite_t1}, index=[0])
        dfs.append(cur_df)

    return pd.concat(dfs)

if __name__ == "__main__":
    data = pd.read_csv(r'../../data/full_data.csv')
    # workaround for weird edge-case when -1 is some reason stays
    for col in ['games_score_winner', 'games_score_loser', 'pts_score_winner', 'pts_score_loser']:
        data[col] = data[col].apply(lambda x: np.NaN if x == -1 else x)
    scores = player_score(data)
    for f in [assign_treatment_L, assign_treatment_diff]:
        print(f.__name__)
        res = Match(data, scores, treatment_func=f)
        res = res.dropna()
        ATE = np.mean(res[['ITE_T=1', 'ITE_T=0']].values.flatten())
        print(ATE)