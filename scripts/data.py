import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re


def get_date(val):
    """
    extract date from date string
    """
    if pd.isna(val):
        return val
    val = str(val)
    try:
        date = datetime.strptime(val, '%Y%m%d').strftime('%m/%d/%Y')
    except ValueError:
        return val
    return date.split('/')[2]


def load_data(males_case=True):
    """
    :param males_case - which data directory to access
    load and prepare dataset located in data folder
    """
    if males_case:
        matches_1 = glob.glob(r'../data/atp_matches_1*.csv')
        matches_2 = glob.glob(r'../data/atp_matches_2*.csv')
    else:
        matches_1 = glob.glob(r'../women_data/wta_matches_1*.csv')
        matches_2 = glob.glob(r'../women_data/wta_matches_2*.csv')
    matches = matches_1 + matches_2

    df = pd.concat([pd.read_csv(f) for f in tqdm(matches)], sort=False)  # keep the original col order
    df['year'] = df['tourney_date'].apply(lambda x: int(get_date(x)))
    df['delta_rank'] = df['winner_rank'] - df['loser_rank']

    return df


def filter_and_normalize(df, thres=500, eps=10**-5):
    """
    normalize delta rank to [0,1] + eps for success score calculation.
    first normalize to [-1,1] and then transform to [0,1].
    filter rank diff > 500 (ignore right tail)
    :param df: data frame as received from load data
    :param thres: value boundaries
    :param eps: epsilon
    :return:
    """
    # fit values to [-500,500]
    df['delta_rank'] = df['delta_rank'].apply(lambda x: -thres if x < -thres else x)
    df['delta_rank'] = df['delta_rank'].apply(lambda x: thres if x > thres else x)
    df['normed_delta_rank'] = df['delta_rank'] / thres
    # 1-rank to penalize high winner rank vs. low looser rank and to reward
    # low winner rank vs. high looser rank
    df['normed_delta_rank'] = 1 - ((1+df['normed_delta_rank'])/2)
    df['normed_delta_rank'] += eps  # add epsilon to avoid obs zero success score
    return df


def naive_success_score(score, rank_factor, iswinner=True, eps = 10**-5):
    """
    calculates the naive score by games ratio
    :param score: score in string format of games won
    :param rank_factor: normed_delta_rank factor from the original data frame.
    """
    if pd.isna(score):
        return -1
    scores = [s.split('(')[0] for s in score.split()]
    winner = 0
    loser = 0
    for games in scores:
        try:
            w,l = re.findall(r'\d+', games)
            winner += int(w)
            loser += int(l)
        except:
            pass
    if winner + loser == 0:
        return None
    # take maximum of loser, 1 to avoid division by zero
    # rank factor will not change between loser\winner score
    if iswinner:
        return  np.log((winner/max(1,loser)) + eps) * rank_factor
    else:
        return np.log((loser / max(1, winner)) + eps) * rank_factor


def ptc_based_success_score(row, iswinner=True, eps=10**-5):
    """
    calculates the point based score
    :param row: data frame row
    """
    winner = row['w_ace'] + row['w_1stWon'] + row['w_2ndWon'] + \
                 (row['l_svpt'] - (row['l_ace'] + row['l_1stWon'] + row['l_2ndWon']))
    loser = row['l_ace'] + row['l_1stWon'] + row['l_2ndWon'] + \
                 (row['w_svpt'] - (row['w_ace'] + row['w_1stWon'] + row['w_2ndWon']))
    if pd.isna(winner) or pd.isna(loser):
        return None
    # take maximum of loser, 1 to avoid division by zero
    if iswinner:
        return np.log((winner/max(1,loser)) + eps) * row['normed_delta_rank']
    else:
        return np.log((loser / max(1, winner)) + eps) * row['normed_delta_rank']


def generate_scores(df):
    """
    calculate success score by both metrics
    """
    df['games_score_winner'] = df.apply(lambda row: naive_success_score(row['score'], row['normed_delta_rank']), axis=1)
    df['games_score_loser'] = df.apply(lambda row: naive_success_score(row['score'], row['normed_delta_rank'],
                                                                       iswinner=False), axis=1)
    df['pts_score_winner'] = df.apply(lambda row: ptc_based_success_score(row), axis=1)
    df['pts_score_loser'] = df.apply(lambda row: ptc_based_success_score(row, iswinner=False), axis=1)
    return df


if __name__ == "__main__":
    males_case = False
    if males_case:
        file_path = r'../data/full_data.csv'
    else:
        file_path = r'../women_data/full_data.csv'
    if not os.path.exists(file_path):
        print("Loading data...")
        df = load_data(males_case)
        print("Data loaded, Filter and Normalize...")
        normalized = filter_and_normalize(df)
        print("Done normalizing, fitting success scores...")
        scored = generate_scores(normalized)
        print("Done!!")
        scored.to_csv(file_path, index=False)
    else:
        scored = pd.read_csv(file_path)















