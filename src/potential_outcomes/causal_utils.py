import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from tqdm import tqdm
import os

score_columns = ['games_score_winner', 'games_score_loser', 'pts_score_winner', 'pts_score_loser']


def _make_preprocessing_pipeline():
    encoder = Pipeline([
        ("encoding", ColumnTransformer([
            ("tourney_level", OrdinalEncoder(), ["tourney_level"]),
            ("round", OrdinalEncoder(), ["round"]),
        ], remainder="passthrough"))
    ])
    return encoder


def factorize_data_for_learning(data):
    cols_to_drop = ['tourney_id', 'draw_size', 'tourney_date', 'match_num', 'winner_entry', 'winner_name',
                    'loser_entry', 'loser_name', 'winner_seed', 'loser_seed', 'delta_rank', 'normed_delta_rank', ]
    data.drop(columns=cols_to_drop, inplace=True)

    ordinal_cols = ['tourney_level', 'round']

    label = LabelEncoder()
    factorzied = pd.DataFrame(columns=data.columns)

    cols_2_consider = data.select_dtypes(np.object).columns.to_list()

    # Now apply the transformation to all the columns:
    for col in data.columns:
        #     print(col)
        if col not in cols_2_consider:
            factorzied[col] = data[col]
            continue
        if data[col].dtype in [np.float64, np.int64]:
            fillval = np.inf
        else:
            fillval = 'NaN'

        if col in ordinal_cols:
            factorzied[col] = data[col]
        else:
            factorzied[col] = label.fit_transform(data[col].fillna(fillval))

    # factorize the ordinal columns separatly (reshape wired bug?!)
    pipeline = _make_preprocessing_pipeline()
    X = pipeline.fit_transform(factorzied)
    columns = ['tourney_level', 'round'] + [c for c in factorzied.columns if c not in ['tourney_level', 'round']]
    # return to a dataframe structure
    factorized_df = pd.DataFrame(X, columns=columns)

    # filter NA and keep only columns with game statistics
    factorized_df_with_stats = factorized_df[factorized_df.year > 1990]
    factorized_df_no_nans = factorized_df_with_stats.dropna(axis=1, thresh=factorized_df_with_stats.shape[0] * 0.8)
    factorized_df_no_nans_at_all = factorized_df_no_nans.dropna()

    return factorized_df_no_nans_at_all


def S_learner(data, y_col='score', model=LinearRegression(), k=10):
    """
    implementation of S learner algorithm
    :param data: input dataframe including treatment vector
    :param y_col: column name in data to
    :param model: LM model. default linear regression
    :param k: cv folds. default 10
    :return: ATE
    """
    preds = []
    X = data[set(data.columns)-set([y_col])]
    y = data[y_col]

    # use cross validation for training
    cv = KFold(k)

    for train, test in cv.split(X, y):
        X_test = X.iloc[test]
        # train on all data
        reg = model.fit(X.iloc[train], y.iloc[train])

        # predict for case and control
        X_treated = X_test.copy()
        X_treated['T'] = 1
        predicted_y1 = reg.predict(X_treated)

        X_control = X_test.copy()
        X_control['T'] = 0
        predicted_y0 = reg.predict(X_control)

        preds.append(predicted_y1 - predicted_y0)

    return np.mean(np.hstack(preds))


if __name__ == "__main__":
    data = pd.read_csv(r'../../data/full_data.csv')
    out_dir = r'../../outputs/'

    mdls = [
        ("LR", Pipeline([
            ("std", StandardScaler()),
            ("classifier", LinearRegression())
        ])),
        ("LR_no_std", LinearRegression()),
        ("Ridge_no_std", Ridge(max_iter=100000)),
        ("RF", RandomForestRegressor()),
        ("Ridge", Pipeline([
            ("std", StandardScaler()),
            ("classifier", Ridge(max_iter=100000))
        ]))
    ]

    factorized_data = factorize_data_for_learning(data)
    ate_dict = {}
    for name, mdl in mdls:
        print(name)
        for score_column in tqdm(score_columns):
            curr_data = factorized_data.copy()
            # assign T only for case when hand are different since S-learner works per match and not per player
            curr_data['T'] = curr_data.apply(lambda row: row['winner_hand'] != row['loser_hand'], axis=1)
            curr_data['Y'] = curr_data[score_column]
            curr_data.drop(columns=score_columns, inplace=True)
            curr_mdl = clone(mdl)
            ate = S_learner(curr_data, 'Y', curr_mdl)
            ate_dict[name, score_column] = ate

    lst = []
    for (mdl_name, col), ate in ate_dict.items():
        lst.append((mdl_name, col, ate))
    learn_df = pd.DataFrame(lst, columns=['model', 'score', 'ATE'])
    learn_df.to_csv(os.path.join(out_dir,'SLEARNER.csv'), index=False)