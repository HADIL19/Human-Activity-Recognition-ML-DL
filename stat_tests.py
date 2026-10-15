"""
stat_tests.py

Statistical tests for the repeated-split experiments.

1. compare_models(): paired tests between models across the repeated runs
   (Wilcoxon signed-rank + Nadeau-Bengio corrected resampled t-test,
   Holm-adjusted, plus a Friedman test over all models).
2. chance_level_test(): permutation test answering
   "is the performance better than chance?".

Important: repeated random splits reuse the same windows, so runs are NOT
independent. Plain paired t-tests give p-values that are too small; the
corrected t-test accounts for this. Report p-values with this caveat.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, permutation_test_score


def _holm(p_values):
    """Holm step-down adjustment of a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def compare_models(results, metric="balanced_accuracy"):
    """
    Paired comparison of models over repeated runs.

    Parameters
    ----------
    results : DataFrame with columns run, model, <metric>, n_train, n_test
    metric  : metric used for the comparison

    Returns
    -------
    pairwise : DataFrame (one row per model pair)
    friedman : dict with statistic and p-value (None if < 3 models)
    """
    wide = results.pivot(index="run", columns="model", values=metric)
    sizes = results.drop_duplicates("run").set_index("run")
    ratio = float((sizes["n_test"] / sizes["n_train"]).mean())

    rows = []
    for a, b in combinations(wide.columns, 2):
        diff = (wide[a] - wide[b]).to_numpy(dtype=float)
        runs = len(diff)

        # Wilcoxon signed-rank (fails when every difference is zero).
        try:
            w_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w_p = 1.0

        # Nadeau & Bengio (2003) corrected resampled t-test.
        var = float(np.var(diff, ddof=1)) if runs > 1 else 0.0
        if var == 0:
            t_p = 1.0 if np.mean(diff) == 0 else 0.0
        else:
            t_stat = np.mean(diff) / np.sqrt((1 / runs + ratio) * var)
            t_p = float(2 * stats.t.sf(abs(t_stat), runs - 1))

        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "mean_a": float(wide[a].mean()),
                "mean_b": float(wide[b].mean()),
                "mean_difference": float(np.mean(diff)),
                "wilcoxon_p": w_p,
                "corrected_t_p": t_p,
            }
        )

    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["wilcoxon_p_holm"] = _holm(pairwise["wilcoxon_p"])
        pairwise["corrected_t_p_holm"] = _holm(pairwise["corrected_t_p"])

    friedman = None
    if wide.shape[1] >= 3:
        stat, p = stats.friedmanchisquare(
            *[wide[c].to_numpy() for c in wide.columns]
        )
        friedman = {
            "metric": metric,
            "statistic": float(stat),
            "p_value": float(p),
        }

    return pairwise, friedman


def chance_level_test(
    models,
    X,
    y,
    n_permutations=100,
    cv_splits=5,
    random_state=42,
):
    """
    Permutation test against chance level.

    Labels are shuffled n_permutations times; the p-value is the share of
    shuffled runs scoring at least as high as the real labels:
        p = (C + 1) / (n_permutations + 1)
    The smallest possible p-value is therefore 1 / (n_permutations + 1).
    """
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    rows = []
    for name, model in models.items():
        score, perm_scores, p_value = permutation_test_score(
            clone(model),
            X,
            y,
            cv=cv,
            scoring="balanced_accuracy",
            n_permutations=int(n_permutations),
            n_jobs=-1,
            random_state=random_state,
        )
        rows.append(
            {
                "model": name,
                "balanced_accuracy": float(score),
                "chance_mean": float(np.mean(perm_scores)),
                "chance_max": float(np.max(perm_scores)),
                "n_permutations": int(n_permutations),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)
