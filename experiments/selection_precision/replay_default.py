"""Compare native search before and after selection precision became configurable.

Run from an OKAPI checkout, or pass --worktree /path/to/OKAPI. The checkout
must contain the pinned historical commit. NumPy and CPU PyTorch are required.
The archive, fixture, and three trajectory logs are temporary.
"""

import argparse
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

BASE_COMMIT = '630fd1f6a12d269c8bc8cd0ebbe0025e0421c302'
REFERENCE_DIGEST = 'a58059775260641eaf48a8a72bcf558431a45aae4af20a5214f3353a9a1de8c3'


def program_state(tree):
    paths = {}

    def visit(node, path):
        paths[id(node)] = path
        for index, child in enumerate(node.children):
            visit(child, (*path, index))

    visit(tree.root, ())
    return (
        repr(tree.structural_key()), tree.mutation_chance,
        [paths[id(node)] for node in tree.value_nodes],
        [paths[id(node)] for node in tree.op_nodes],
    )


def snapshot(engine, count, evaluated):
    state = engine.rng.get_state()
    rng_bytes = state[0].encode('ascii') + state[1].astype('<u4', copy=False).tobytes() + struct.pack('<IId', *state[2:])
    programs = [program_state(tree) for tree in engine.population]
    return [
        hashlib.sha256(repr(programs).encode('utf-8')).hexdigest(),
        list(engine.fitnesses.shape),
        hashlib.sha256(engine.fitnesses.astype('<f8', copy=False).tobytes()).hexdigest(),
        hashlib.sha256(rng_bytes).hexdigest(), count, evaluated.hexdigest(),
    ]


def save_inputs(directory, backend):
    import torch

    directory.mkdir(exist_ok=True)
    rng = np.random.RandomState(118792)
    labels = rng.randint(2, size=(173, 1)).astype(np.float64)
    predictions = np.clip(.03 + .94 * (.31 * labels + .69 * rng.rand(6, 173, 1)), 0, 1)

    def write(name, values):
        if backend == 'numpy':
            path = directory / f'{name}.npy'
            np.save(path, values)
        else:
            path = directory / f'{name}.pt'
            torch.save(torch.from_numpy(values), path)
        return path

    return [write(f'm{i}', p) for i, p in enumerate(predictions)], write('label', labels)


def run_arm(providers, labels, *, backend, seed, reuse, mode):
    from okapi import Okapi
    from okapi.pareto import maximize

    evaluations = hashlib.sha256()
    calls = 0

    def fitness(prediction, target):
        nonlocal calls
        value = float(1 - np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))
        evaluations.update(struct.pack('<d', value))
        calls += 1
        return value

    options = {} if mode == 'default' else {'selection_fitness_decimals': int(mode)}
    engine = Okapi(
        providers, labels, 6, 2, 2, backend=backend, seed=seed,
        objective_functions=(fitness,), objectives=(maximize,), reuse_survivor_fitness=reuse,
        operator_point_mutation=True, **options,
    )
    trace = []
    for _ in range(18):
        engine.run_iteration()
        trace.append(snapshot(engine, calls, evaluations))
    return trace


def worker(source, mode, output):
    # Each version is imported in a fresh process so the editable installation cannot override it.
    sys.path.insert(0, str(source))
    import torch
    from loguru import logger

    okapi = importlib.import_module('okapi')
    logger.remove()
    if Path(okapi.__file__).resolve().parent != source / 'okapi':
        raise RuntimeError(f'OKAPI imported from an unexpected path: {okapi.__file__}')
    torch.set_num_threads(1)
    arms = {}
    for backend in ('numpy', 'torch'):
        providers, labels = save_inputs(output.parent / backend, backend)
        for seed in (799, 816):
            for reuse in (False, True):
                arms[f'{backend}-{seed}-{reuse}'] = run_arm(providers, labels, backend=backend, seed=seed, reuse=reuse, mode=mode)
    output.write_text(json.dumps(arms, indent=2) + '\n', encoding='utf-8')


def run(worktree, scratch):
    historical = scratch / 'base'
    historical.mkdir()
    archive = scratch / 'base.tar'
    subprocess.run(['git', '-C', str(worktree), 'archive', f'--output={archive}', BASE_COMMIT], check=True)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(historical)], check=True)
    versions = [('previous', historical, 'default'), ('current-default', worktree, 'default'), ('current-3', worktree, '3')]
    traces = []
    environment = {
        **os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1', 'MPLCONFIGDIR': str(scratch / 'mpl'),
    }
    for name, source, mode in versions:
        output = scratch / f'{name}.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(source), mode, str(output)], check=True, env=environment)
        traces.append(output.read_bytes())
    if not traces[0] == traces[1] == traces[2]:
        raise AssertionError('Saved program, objective, fitness or random-state traces differ')
    digest = hashlib.sha256(traces[0]).hexdigest()
    arms = json.loads(traces[0])
    print(f'{sum(map(len, arms.values()))} identical generation checkpoints per comparison')
    print(f'{sum(trace[-1][-2] for trace in arms.values())} objective calls per version')
    print(f'Trajectory SHA-256 {digest}')
    if digest != REFERENCE_DIGEST:
        raise AssertionError('Within-environment parity held, but the recorded reference trajectory differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, default=Path.cwd(), help='Current OKAPI checkout, default current directory')
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        source, mode, output = args.worker
        worker(Path(source).resolve(), mode, Path(output).resolve())
        return
    with TemporaryDirectory(prefix='okapi-default-precision-') as tmp:
        run(args.worktree.resolve(), Path(tmp))


if __name__ == '__main__':
    main()
