from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import numpy as np
from scipy.stats import norm
import pandas as pd

def configs_to_dataframe(configs, features):
    df = pd.DataFrame(columns=features)
    for config in configs:
        df.loc[len(df)] = [config[feat] for feat in features]
    return df

def compute_ei(space, data, n_challengers = 1000):
    features = data.drop(columns=["seed", "identifier", "accuracy", "firing_rate", "balance"])
    surrogate_acc = GPR()
    surrogate_acc.fit(features, data["accuracy"])
    
    opt_acc = np.max(surrogate_acc.predict(features))
    
    challengers = space.sample_configuration(n_challengers)
    challengers_df = configs_to_dataframe(challengers, features.columns)
    
    mean, std = surrogate_acc.predict(challengers_df, return_std=True)
    
    Z = (mean - opt_acc) / std
    ei = (mean - opt_acc) * norm.cdf(Z) + std * norm.pdf(Z)

    return ei, challengers_df, challengers, surrogate_acc
        
def compute_constrained_ei(data, challengers_df, ei_vals, balance_constraint, fr_constraint):
    surrogate_balance, surrogate_fr = GPR(), GPR()
    
    features = data.drop(columns=["seed","identifier", "accuracy", "firing_rate", "balance"])
    
    surrogate_balance.fit(features, data["balance"])
    surrogate_fr.fit(features, data["firing_rate"])
    
    balance_mean = surrogate_balance.predict(challengers_df, return_std=False)
    fr_mean = surrogate_fr.predict(challengers_df, return_std=False)
    
    valid = np.logical_and(balance_mean > balance_constraint, fr_mean < fr_constraint)
    
    return ei_vals * valid, challengers_df, surrogate_balance, surrogate_fr

def sample_constrained_bayesian(space, data, balance_constraint, fr_constraint, n_challengers = 1000):
    ei_vals, challengers_df, challengers, _ = compute_ei(space, data, n_challengers)
    ei_vals, challengers_df, _, _ = compute_constrained_ei(data, challengers_df, ei_vals, balance_constraint, fr_constraint)
    
    return challengers[np.argmax(ei_vals)]