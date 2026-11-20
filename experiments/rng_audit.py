"""Run with PYTHONPATH pointing to a checkout of OKAPI."""
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from loguru import logger
from okapi.okapi import Okapi

logger.disable("okapi")

def engine(data, labels, seed):
    return Okapi(sorted(data.iterdir()), labels, 5, 3, 3, seed=seed, backend="numpy",
                 objective_functions=(lambda pred, gt: 1 - np.mean((pred - gt) ** 2),))

def trajectory(data, labels, interleaved):
    opt = engine(data, labels, 7)
    other = engine(data, labels, 91) if interleaved else None
    hashes = []
    for _ in range(8):
        if other:
            other.run_iteration()
            np.random.random(31)
        opt.run_iteration()
        state = hashlib.sha256()
        for tree, fitness in zip(opt.population, opt.fitnesses):
            state.update(repr(tree).encode())
            state.update(tree.predict().tobytes())
            state.update(fitness.tobytes())
        hashes.append(state.hexdigest())
    return hashes

with TemporaryDirectory() as tmp:
    data = Path(tmp) / "preds"
    data.mkdir()
    rng = np.random.RandomState(711)
    for i in range(6):
        np.save(data / f"m{i}.npy", rng.rand(60, 1))
    labels = Path(tmp) / "gt.npy"
    np.save(labels, rng.randint(2, size=(60, 1)))
    print(json.dumps({key: trajectory(data, labels, flag) for key,flag in (("alone",False),("interleaved",True))}))
