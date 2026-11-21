"""Summarize first survival and full-program recurrence from a frozen GP event ledger."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fusionbench.eval.gate_necessity import native_tree


def summarize(source, output):
    events = pd.read_csv(source / "events.csv.gz")
    runs = pd.read_csv(source / "runs.csv.gz")
    arms = ["type_only", "both"]
    keys = ["data_seed", "fit_law", "gp_seed", "arm"]
    # All submitted opportunities in complete selections before the same fit-call ceiling.
    submitted = events.loc[
        events.arm.isin(arms) & events.offspring_index.notna() & events.selection_call.le(1800)
    ].copy()
    assert submitted.changed.all() and submitted["survived"].notna().all()
    # Dummy tensors suffice for reconstructing a structural key. No data or labels feed the audit.
    tensors = {f"p{i}.npy": np.zeros((1, 1)) for i in range(6)}
    wins = runs.loc[
        runs.method.isin(arms) & runs.checkpoint.eq("calls_1800") & runs.decision.eq("fit")
        & runs.split.eq("iid") & runs.test_law.eq(runs.fit_law)
    ].rename(columns={"method": "arm"})
    assert not wins.duplicated(keys).any() and len(wins) == 12 * 2 * 2 * len(arms)

    def program_key(program):
        tree = native_tree(json.loads(program), tensors)
        return hashlib.sha256(repr(tree.structural_key()).encode()).hexdigest()

    wins["winner_key"] = wins.program.map(program_key)
    joined = submitted.merge(wins[keys + ["winner_key"]], on=keys, how="left", validate="many_to_one")
    assert joined.winner_key.notna().all()
    joined["same_as_winner"] = joined.evaluated & joined.new_sha256.eq(joined.winner_key)
    joined["survived_and_same_as_winner"] = joined.survived & joined.same_as_winner
    joined["births"] = 1
    measures = ["births", "evaluated", "survived", "same_as_winner", "survived_and_same_as_winner"]
    cols = ["fit_law", "arm", "after_type"]
    by_bank = joined.groupby([*cols, "data_seed"])[measures].sum().reset_index()
    for numerator, denominator in (("evaluated", "births"), ("survived", "evaluated"), ("survived", "births")):
        by_bank[f"{numerator}_per_{denominator}"] = by_bank[numerator] / by_bank[denominator].replace(0, np.nan)
    counts = by_bank.groupby(cols)[measures].sum()
    rates = by_bank.groupby(cols)[["evaluated_per_births", "survived_per_evaluated", "survived_per_births"]].mean()
    output.mkdir(parents=True, exist_ok=True)
    counts.join(rates).reset_index().to_csv(output / "summary.csv", index=False)
    by_bank.to_csv(output / "by_bank.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory with organic-composition runs and events")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output)
