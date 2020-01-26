from causal_utils import *
from misc_utils import factorize_non_numeric_values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import clone

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_val_score


def est_propensity(data):
    """
    estimate propensity score by multiple classifiers
    """
    features = [c for c in data.columns if 'x_' in c]
    clfs = [
        ("SVC-linear", SVC(kernel="linear", C=0.025, probability=True)),
        ("RF", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
        ("NN", MLPClassifier(alpha=1, max_iter=1000)),
        ("Adaboost", AdaBoostClassifier()),
        ("LR", LogisticRegression(max_iter=10000)),
        ("LR-no_reg", LogisticRegression(C=np.Inf, max_iter=100000)),
    ]
    propensity_dict = {}
    for name, clf in clfs:
        curr_clf = Pipeline([
            ("std", StandardScaler()),
            ("classifier", clone(clf))
        ])
        aucs = cross_val_score(curr_clf, data[features], data['T'], cv=10, scoring="roc_auc")
        pscore = get_propensity(data, features=features, mdl=curr_clf)
        propensity_dict[name] = pscore, np.mean(aucs), np.std(aucs)
        print(name, propensity_dict[name][1], propensity_dict[name][2])
    return propensity_dict


def IPW_ATT(data):
    return ipw(data)


def Matching_ATT(data, delta=100):

    calipers = np.linspace(0, 0.5, num=11)
    features = [c for c in data.columns if 'x_' in c]
    metrics = ['euclidean', 'manhattan', 'chebyshev','mahalanobis']

    # calculate invert cov matrix for mahalanobis
    cov = np.cov(data[features].T)
    IV = np.linalg.inv(cov)

    att_dict = {}
    for c in calipers:
        for metric in metrics:
            curr_data = data.copy()
            matched = Match(curr_data, features, dist_metric=metric, caliper=c, delta=delta,IV=IV)
            att = get_matching_att(curr_data, matched)
            att_dict[metric, c] = att

        # now test using correlations
        curr_data = data.copy()
        matched = Match(curr_data, features, dist_metric='corr', caliper=c, delta=delta, corr_mode=True)
        att = get_matching_att(curr_data, matched)
        att_dict['correlation', c] = att
    return att_dict


def S_learner_ATT(data, k=10):
    features = [c for c in data.columns if 'x_' in c]
    mdls = [
        ("LR", Pipeline([
            ("std", StandardScaler()),
            ("classifier", LinearRegression())
        ])),
        ("RF", RandomForestRegressor()),
        ("GBR", GradientBoostingRegressor()),
        ("Ridge", Pipeline([
            ("std", StandardScaler()),
            ("classifier", Ridge(max_iter=100000))
        ]))
    ]

    att_dict = {}
    for name, mdl in mdls:
        curr_data = data.copy()
        curr_mdl = clone(mdl)
        att = S_learner(curr_data, features, curr_mdl, k)
        att_dict[name] = att
    return att_dict


def T_learner_ATT(data, k=10):
    features = [c for c in data.columns if 'x_' in c]
    mdls = [
        ("LR", Pipeline([
            ("std", StandardScaler()),
            ("classifier", LinearRegression())
        ])),
        ("LR_no_scale", LinearRegression()),
        ("RF", RandomForestRegressor()),
        ("GBR", GradientBoostingRegressor()),
        ("Ridge", Pipeline([
            ("std", StandardScaler()),
            ("classifier", Ridge(max_iter=100000))
        ])),
        ("Ridge_no_scale", Ridge(max_iter=100000)),
    ]

    att_dict = {}
    for name, mdl in mdls:
        for name2, mdl2 in mdls:
            curr_data = data.copy()
            curr_mdl = clone(mdl)
            curr_mdl2 = clone(mdl2)
            att = T_learner(curr_data, features, curr_mdl, curr_mdl2, k)
            att_dict[name, name2] = att
    return att_dict


def find_best_model(file_path):
    """
    run each ATT evaluation metric and save best models
    :param file_path: a file path to a dataframe
    :return:
    """
    data = pd.read_csv(file_path, index_col='Unnamed: 0')
    data = factorize_non_numeric_values(data)

    res_d = dict()

    propensities = est_propensity(data)
    for key in propensities:
        data['Propensity'] = propensities[key][0]

        # ipw
        att_ipw = IPW_ATT(data)
        print("Propensity {}, ipw ATT {}".format(key, att_ipw))
        res_d.setdefault(key, dict())['IPW'] = att_ipw

        # matching
        att_matching = Matching_ATT(data)
        res_d.setdefault(key, dict())['Matching'] = att_matching
        print("Propensity {}, matching ATT {}".format(key, att_matching))

    # learners are independent of propensity score
    att_slearner = S_learner_ATT(data)
    print("S learner ATT ", att_slearner)
    res_d['S_learner'] = att_slearner

    att_tlearner = T_learner_ATT(data)
    print("T learner ATT ", att_tlearner)
    res_d['T_learner'] = att_tlearner

    return res_d

def get_results(file_paths):
    """
    generate the propensity and ATT scores based on best models according to simulations
    :param file_path: a file path to a dataframe
    :return:
    """
    prop_lst = []
    ATT_dict = dict()
    for idx, file_path in enumerate(file_paths):
        ATT_dict[idx] = dict()
        data = pd.read_csv(file_path, index_col='Unnamed: 0')
        data = factorize_non_numeric_values(data)
        features = [c for c in data.columns if 'x_' in c]

        # get propensity score
        prop_clf = Pipeline([
            ("std", StandardScaler()),
            ("LR", clone(LogisticRegression(max_iter=10000)))
        ])
        pscore = get_propensity(data, features=features, mdl=prop_clf)
        prop_lst.append(pscore)

        # get ATT IPW estimator
        data['Propensity'] = pscore

        # ipw
        att_ipw = IPW_ATT(data)
        ATT_dict[idx]['ipw'] = att_ipw

        # Matching
        curr_data = data.copy()
        matched = Match(curr_data, features, dist_metric='chebyshev', caliper=0.15)
        matching_att = get_matching_att(data, matched)
        ATT_dict[idx]['Matching'] = matching_att

        # S-learner
        mdl = Pipeline([
            ("std", StandardScaler()),
            ("classifier", Ridge(max_iter=100000))
        ])
        curr_data = data.copy()
        curr_mdl = clone(mdl)
        s_learner_att = S_learner(curr_data, features, curr_mdl)
        ATT_dict[idx]['S Learner'] = s_learner_att

        # T-learner
        curr_data = data.copy()
        treated_mdl = Ridge(max_iter=100000)
        control_mdl = LinearRegression()
        t_learner_att = T_learner(curr_data, features, treated_mdl, control_mdl)
        ATT_dict[idx]['T Learner'] = t_learner_att

    return prop_lst, ATT_dict





