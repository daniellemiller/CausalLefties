from sklearn.linear_model import Ridge

from misc_utils import factorize_non_numeric_values
from potential_outcomes.causal_utils import T_learner
import pandas as pd


data = pd.read_csv(r'../data/full_data.csv')
features = [c for c in data.columns if 'hand' not in c or 'score' not in c]

data = data.dropna(subset=['games_score'])
data['T'] = data.apply(lambda row: 1 if (row['winner_hand'] =='L') & (row['loser_hand'] == 'R') else 0, axis=1)
data['Y'] = data['games_score']
#data = factorize_non_numeric_values(data)
data.reset_index(drop=True, inplace=True)

res = T_learner(data, features, model_treated=Ridge(max_iter=100000), model_control=Ridge(max_iter=100000), k=1)
print(res)