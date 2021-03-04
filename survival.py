from pprint import pprint
import pickle

from harmoniums import SurvivalHarmonium
from harmoniums.utils import double_cross_validate
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform

from load import load_dataset


random_state = 1234

X, y = load_dataset(granularity="chromosome")
X = X.join(y.drop(columns=["clinical_response", "response_grouped", "pfs>1yr"]))

survival_columns = ["os_months", "pfs_months"]
event_columns = ["os_event", "pfs_event"]
numeric_columns = ["normalized_tmb_t0", "normalized_tmb_t1"]
categorical_columns = sorted(
    set(X.columns) - set(survival_columns) - set(event_columns) - set(numeric_columns)
)


def concordance_score(estimator, X, y=None):
    if not isinstance(estimator, SurvivalHarmonium):
        estimator = estimator.best_estimator_
    return estimator.concordance_index(X, conditional_probability=False,)


def conditional_concordance_score(estimator, X, y=None):
    if not isinstance(estimator, SurvivalHarmonium):
        estimator = estimator.best_estimator_
    return estimator.concordance_index(X, conditional_probability=True)


time_horizon = X[survival_columns].max(axis=0).values
harmonium = SurvivalHarmonium(
    categorical_columns=categorical_columns,
    survival_columns=survival_columns,
    event_columns=event_columns,
    numeric_columns=numeric_columns,
    verbose=True,
    log_every_n_iterations=None,
    time_horizon=time_horizon,
    risk_score_time_point="median",
    metrics=tuple(),
    CD_steps=1,
    n_hidden_units=1,
)
harm_hyperparams = {
    "learning_rate": loguniform(1e-7, 0.005),
    "n_epochs": loguniform(1e1, 1e5),
    "momentum_fraction": uniform(0, 1.0),
    "mini_batch_size": loguniform(25, 1000),
    "weight_decay": loguniform(1e-5, 0.1),
    "persistent": [True, False],
    "guess_weights": [True, False],
}

s = double_cross_validate(
    harmonium,
    X,
    None,
    harm_hyperparams,
    m=5,
    n=5,
    scoring={
        "conditional_concordance": conditional_concordance_score,
        "concordance": concordance_score,
    },
    refit="conditional_concordance",
    n_iter=25,
    n_jobs=-1,
    random_state=random_state,
)

pickle_location = "models/harmonium.pickle"
estimator_fold = []
for fold_number, est in enumerate(s["estimator"]):
    model = getattr(est, "best_estimator_", None)
    params = getattr(est, "best_params_", None)
    score = getattr(est, "best_score_", None)
    estimator_fold.append({"estimator": model, "params": params, "score": score})
    print(f"Fold {fold_number+1}")
    print(f"Best params:")
    pprint(params)
    print("Best score:", score)
    print("==" * 10)

data = {
    "folds": estimator_fold,
}
with open(pickle_location, mode="wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
