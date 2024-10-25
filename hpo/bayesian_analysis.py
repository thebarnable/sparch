from bayesian import *
from hpo import get_configspace
import pandas as pd

results = pd.read_csv("/mnt/data40tb/sparch_hpo/opt3/results.csv")
configspace = get_configspace("opt3", 42)

BALANCE_CONSTRAINT = 0.03
FR_CONSTRAINT = 0.00015

print("Computing Expected Improvment values...")
ei, challengers, _,  surrogate_acc = compute_ei(configspace, results, 10000)
print("Compute Constrained Expected Improvment values...")
constrained_ei, challengers, surrogate_balance, surrogate_fr = compute_constrained_ei(results, challengers, ei, BALANCE_CONSTRAINT, FR_CONSTRAINT)

print("Best configuration:")
best_indices = np.argsort(constrained_ei)[:10]
best_challengers_feat = challengers.iloc[best_indices]
best_challengers = best_challengers_feat.copy()
best_challengers.insert(len(best_challengers.columns), "accuracy_pred", surrogate_acc.predict(best_challengers_feat))
best_challengers.insert(len(best_challengers.columns), "balance_pred", surrogate_balance.predict(best_challengers_feat))
best_challengers.insert(len(best_challengers.columns), "fr_pred", surrogate_fr.predict(best_challengers_feat))
print(best_challengers)