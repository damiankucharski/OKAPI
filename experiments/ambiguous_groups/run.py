"""Run the registered two-world proposal study from the OKAPI checkout."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.ambiguous_groups.methods import fit_method, matching_code
from experiments.ambiguous_groups.worlds import make_design, sample
from okapi.globals import set_backend, set_postprocessing_function

ARMS = ("top_one", "top_two", "random_two", "gp", "mean", "stack")


def structural_reference(x, paths):
    positive = x[:, :, 1]
    path_values = []
    for left, right in paths:
        path_values.append(np.minimum(positive[:, left].mean(axis=1), positive[:, right].mean(axis=1)))
    return np.maximum(*path_values)


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def run_bank(bank):
    design_seed, search_seed = [int(np.random.SeedSequence([bank, tag]).generate_state(1)[0]) for tag in (91, 92)]
    design = make_design(design_seed)
    fit_x, fit_labels = sample(design, bank, 0, 384, .60)
    val_x, val_labels = sample(design, bank, 1, 256, .60)
    fit_x.flags.writeable = val_x.flags.writeable = False
    predictors = {}
    fits = []
    for world in ("A", "H"):
        order_rng = np.random.default_rng([bank, 111, int(world == "H")])
        for arm in order_rng.permutation(ARMS):
            # No alternative labels, latent assignments, or test rows cross
            # this boundary. The two worlds use one and the same sensor matrix.
            pred, cost = fit_method(arm, fit_x, fit_labels[world], val_x, val_labels[world],
                                    seed=search_seed)
            predictors[world, arm] = pred
            fits.append(dict(bank=bank, world=world, arm=arm, **cost))
    # Privileged reference applies the latent structure to observed scores.
    # Truth is joined only after all twelve independent policy fits finish.
    recovery = []
    for world in ("A", "H"):
        true_code = matching_code(design.matching(world))
        for row in (r for r in fits if r["world"] == world and r["arm"] in ("top_one", "top_two", "random_two")):
            proposals = row["offered_matchings"].split(";")
            recovery.append(dict(bank=bank, world=world, arm=row["arm"], true_matching=true_code,
                                 first_correct=int(proposals[0] == true_code), offered_correct=int(true_code in proposals),
                                 selected_correct=int(row["selected_matching"] == true_code)))
        paths = design.paths(world)
        predictors[world, "known_structure"] = lambda x, paths=paths: structural_reference(x, paths)
    risks = []
    for domain, stream, sigma in (("iid", 2, .60), ("shift", 3, 1.25)):
        x, labels = sample(design, bank, stream, 4096, sigma)
        x.flags.writeable = False
        disagrees = labels["A"] != labels["H"]
        for (world, arm), predict in predictors.items():
            start = perf_counter()
            p = np.asarray(predict(x)).reshape(-1)
            elapsed = perf_counter() - start
            assert p.shape == labels[world].shape and np.all((p >= 0) & (p <= 1))
            y = labels[world]
            risks.append(dict(bank=bank, world=world, arm=arm, domain=domain, n=len(y),
                              disagree_n=int(disagrees.sum()), brier=mse(p, y),
                              accuracy=float(np.mean((p >= .5) == y)), inference_seconds=elapsed,
                              brier_agree=mse(p[~disagrees], y[~disagrees]), brier_disagree=mse(p[disagrees], y[disagrees])))
    return fits, recovery, risks


def contrasts(risks):
    # Both worlds share observations; pair within bank before uncertainty.
    out = []
    for domain, cell in risks.groupby("domain"):
        wide = cell.pivot_table(index="bank", columns="arm", values="brier")
        for control in ("random_two", "top_one", "gp", "stack", "mean", "known_structure"):
            differences = wide["top_two"] - wide[control]
            margin = t.ppf(.975, len(wide) - 1) * differences.sem() if len(wide) > 1 else np.nan
            delta = differences.mean()
            out.append(dict(domain=domain, arm="top_two", control=control, banks=len(wide),
                            delta=delta, lower=delta-margin, upper=delta+margin,
                            wins=int((differences < 0).sum()), ties=int((differences == 0).sum()),
                            losses=int((differences > 0).sum())))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(41200, 41212)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each seed bank must be distinct")
    args.out.mkdir(parents=True, exist_ok=True)
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(lambda p: p)
    fits, recovery, risks = [], [], []
    with threadpool_limits(1):
        for bank in args.seeds:
            a, b, c = run_bank(bank)
            fits.extend(a)
            recovery.extend(b)
            risks.extend(c)
    risks = pd.DataFrame(risks)
    pd.DataFrame(fits).to_csv(args.out / "fits.csv", index=False)
    pd.DataFrame(recovery).to_csv(args.out / "recovery.csv", index=False)
    risks.to_csv(args.out / "test.csv", index=False)
    contrasts(risks).to_csv(args.out / "contrasts.csv", index=False)


if __name__ == "__main__":
    main()
