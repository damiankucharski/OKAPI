import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Type, Union

import numpy as np
import numpy.typing as npt
import tqdm
from loguru import logger

import okapi.lib_types as lib_types
from okapi.backend.backend import Backend
from okapi.callback import Callback
from okapi.crossover import crossover, tournament_selection_indexes
from okapi.fitness import average_precision_fitness
from okapi.globals import BACKEND as B
from okapi.globals import DEVICE, clear_eval_context, set_eval_context, set_postprocessing_function
from okapi.lib_types import Tensor
from okapi.mutation import get_allowed_mutations, mutate_parameters
from okapi.node import OperatorNode
from okapi.operators import CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, WEIGHTED_MEAN
from okapi.pareto import _get_optimal_point_based_on_list_of_objective_functions, maximize
from okapi.population import choose_pareto, choose_pareto_then_proximity, initialize_individuals
from okapi.tree import Tree
from okapi.utils import first_uniques_mask, mark_paths


class Okapi:
    """
    Main class for evolutionary model ensemble optimization.

    Okapi uses genetic programming to evolve tree-based ensembles of machine learning models.
    The algorithm creates a population of trees where each tree represents a different way of
    combining model predictions. Through evolution (crossover and mutation), it searches for
    optimal ensemble structures that maximize a fitness function.

    Each tree has ValueNodes that contain tensor predictions from individual models, and
    OperatorNodes that define how to combine these predictions (e.g., mean, min, max, weighted mean).
    The evolution process selects and combines high-performing trees to produce better ensembles.

    Attributes:
        population_size: Number of individuals in the population
        population_multiplier: Factor determining how many additional trees to generate in each iteration
        tournament_size: Number of trees to consider in tournament selection
        fitness_function: Function used to evaluate the fitness of each tree
        callbacks: Collection of callbacks for monitoring/modifying the evolution process
        allowed_ops: Operator node types allowed in tree construction
        train_tensors: Dictionary mapping model names to their prediction tensors
        gt_tensor: Ground truth tensor for comparison
        population: Current population of trees
        additional_population: Additional trees generated during evolution
    """

    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str, Iterable[Path], Iterable[str]],
        population_size: int,
        population_multiplier: int,
        tournament_size: int,
        minimize_node_count: bool = True,
        objective_functions: Sequence[Callable[[Tree, lib_types.Tensor], float]] = (average_precision_fitness,),
        objectives: Sequence[Callable[[float, float], bool]] = (maximize,),
        allowed_ops: Sequence[Type[OperatorNode]] = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        callbacks: Iterable[Callback] = tuple(),
        backend: Union[str, None] = None,
        seed: int = 0,
        postprocessing_function=None,
        mutation_strength: float = 0.1,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        cv=None,
        cv_penalty: float = 0.0,
    ):
        """
        Initialize the Okapi evolutionary algorithm.

        Args:
            preds_source: Source of model predictions, can be a path to directory or iterable of paths
            gt_path: Path to ground truth data, can be a single path or iterable of paths. Should match preds_source by order
            population_size: Size of the population to evolve
            population_multiplier: Factor determining how many additional trees to generate
            tournament_size: Number of trees to consider in tournament selection
            minimize_node_count: Whether the pareto frontier models should also consider node count.
            objective_functions: Functions that calculate the fitnesses that are to be optimized
            objectives: Functions that copare two fitnesses and return True if first is better than second. Usually maximize or minimize
            allowed_ops: Sequence of operator node types that can be used in trees
            callbacks: Iterable of callback objects for monitoring/modifying evolution
            backend: Optional backend implementation for tensor operations
            seed: Random seed for reproducibility
            postprocessing_function: Function applied after each Op Node.
            Most of the operations may break some data characteristics, for example vector summing to one. This can be used to fix that.
            mutation_strength: Controls magnitude of parameter mutations (default 0.1).
            Higher values cause larger parameter changes during evolution.
            max_depth: Optional hard cap on tree depth (levels; root = 1, a value->op->value
            fusion = 3). ``None`` (default) imposes no limit and reproduces the historical
            behaviour exactly. When set, variation never produces an individual deeper than this.
            max_nodes: Optional hard cap on total node count. ``None`` (default) imposes no limit.
            Both caps are enforced at generation time (mutation + crossover), not merely at
            selection, so they also bound evaluation cost - see ``_within_caps``.
            cv: Cross-validation for the fitness signal, to make selection less prone to
            exploiting a noisy validation estimate. ``None`` / ``1`` (default) keeps the
            single full-split fitness (historical behaviour, exactly). An ``int`` k uses
            ``KFold(k, shuffle=True, random_state=seed)``; any scikit-learn splitter object
            (e.g. ``StratifiedKFold``) is accepted as-is. Each objective then becomes the mean
            of its per-fold scores. (OKAPI operators are pointwise across samples, so this is
            an evaluation-averaging of the fitness over disjoint row folds, not a per-fold
            re-fit: it stabilises the estimate and exposes its variance, see ``fitness_stds``.)
            cv_penalty: When ``cv`` is set, subtract ``cv_penalty * std`` of the per-fold
            scores from each objective (added for minimise objectives), penalising trees whose
            fitness is unstable across folds. ``0.0`` (default) = pure fold mean.
        """
        if backend is not None:
            Backend.set_backend(backend)
        if seed is not None:
            np.random.seed(seed)
        if postprocessing_function:
            set_postprocessing_function(postprocessing_function)

        self.population_size = population_size
        self.population_multiplier = population_multiplier
        self.tournament_size = tournament_size
        self.minimize_node_count = minimize_node_count
        self.mutation_strength = mutation_strength
        self.seed = seed
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.cv_penalty = cv_penalty

        self.objective_functions = objective_functions
        self.objectives = objectives
        assert len(objectives) == len(objective_functions), "The number of (optimization) objectives and objective functions is not the same"
        self.optimal_point = _get_optimal_point_based_on_list_of_objective_functions(self.objectives)

        self.callbacks = callbacks
        self.allowed_ops = allowed_ops

        self.train_tensors, self.gt_tensor = self._build_train_tensors(preds_source, gt_path)
        self.ids, self.models = list(self.train_tensors.keys()), list(self.train_tensors.values())
        self._validate_input()

        # Cross-validated fitness folds (None => single full-split fitness, as before).
        self._cv_folds = self._build_cv_folds(cv)
        self.fitness_stds: None | npt.NDArray[np.float64] = None

        # state
        self.should_stop = False

        self.population = self._initialize_population()
        self.additional_population: List[Tree] = []  # for potential callbacks
        self.fitnesses: None | npt.NDArray[np.float64] = None

    def _call_hook(self, hook_name):
        """
        Call a specific hook on all registered callbacks.

        Args:
            hook_name: Name of the hook to call
        """
        for callback in self.callbacks:
            getattr(callback, hook_name)(self)

    def _initialize_population(self):
        """
        Initialize the population of trees.

        Creates simple trees using available prediction tensors.

        Returns:
            List of initialized Tree objects
        """
        logger.info(f"Initializing population with size {self.population_size}")
        population = initialize_individuals(self.train_tensors, self.population_size)
        logger.debug(f"Population initialized with {len(population)} individuals")
        return population

    def _build_cv_folds(self, cv) -> None | list:
        """Precompute the (disjoint) row index folds for cross-validated fitness, once.

        ``None`` / ``1`` -> ``None`` (single full-split fitness, unchanged). An ``int`` k ->
        ``KFold(k, shuffle=True, random_state=seed)``; any scikit-learn splitter object is
        used as given (e.g. ``StratifiedKFold``, which uses the labels). Returns the list of
        per-fold *test* index arrays partitioning the rows.
        """
        if cv is None or (isinstance(cv, int) and cv <= 1):
            return None
        y = np.asarray(B.to_numpy(self.gt_tensor)).ravel()
        if isinstance(cv, int):
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
        x_dummy = np.zeros((len(y), 1))
        return [np.asarray(test_idx) for _, test_idx in cv.split(x_dummy, y)]

    def _calculate_fitnesses(self, trees: None | List[Tree] = None) -> npt.NDArray[np.float64]:
        """
        Calculate fitness values for the given trees.

        Uses memory-efficient prediction extraction via tree.predict() which
        clears intermediate evaluation caches after each tree is evaluated.

        With cross-validated fitness enabled (``cv`` set) each objective is the mean of its
        per-fold scores (minus ``cv_penalty * std``); the prediction is computed once on the
        full split and sliced per fold (valid because every operator is pointwise across
        samples), so the cost over the single-split path is only the extra cheap objective
        evaluations. Per-fold standard deviations for the trees passed to *this* call are
        stored on ``self.fitness_stds`` (so after a full ``run_iteration`` they correspond to
        the last internal evaluation, not necessarily the trimmed population; call this on a
        tree list to get a view aligned to it).

        Args:
            trees: List of trees to evaluate. If None, uses the current population.

        Returns:
            NumPy array of fitness values
        """
        if trees is None:
            trees = self.population
        logger.debug(f"Calculating fitness for {len(trees)} trees")

        n_obj = len(self.objective_functions)
        fitnesses = np.zeros(shape=(len(trees), n_obj))
        stds = np.zeros(shape=(len(trees), n_obj)) if self._cv_folds is not None else None

        # Supervised operators (e.g. IdealPointTrustNode) read the fit ground truth from the
        # global eval-context during their evaluation; expose it only for this fitness pass and
        # clear it afterwards, so prediction sees no y and falls back to its fit-cached weights.
        set_eval_context(self.gt_tensor)
        try:
            for tree_idx, tree in enumerate(trees):
                # Get prediction with automatic cache clearing for memory efficiency
                prediction = tree.predict(clear_cache=True)

                for obj_idx, objective_function in enumerate(self.objective_functions):
                    if self._cv_folds is None:
                        fitnesses[tree_idx, obj_idx] = objective_function(prediction, self.gt_tensor)
                    else:
                        fold_scores = np.array(
                            [objective_function(prediction[idx], self.gt_tensor[idx]) for idx in self._cv_folds]
                        )
                        mean_score, std_score = float(fold_scores.mean()), float(fold_scores.std())
                        # Penalise across-fold instability in the objective's *worsening* direction.
                        direction = 1.0 if self.objectives[obj_idx] is maximize else -1.0
                        fitnesses[tree_idx, obj_idx] = mean_score - direction * self.cv_penalty * std_score
                        stds[tree_idx, obj_idx] = std_score
        finally:
            clear_eval_context()

        self.fitness_stds = stds
        return fitnesses

    def run_iteration(self):
        """
        Run a single iteration of the evolutionary algorithm.

        This method:
        1. Calculates fitness values for the current population
        2. Performs tournament selection and crossover to create new trees
        3. Applies mutations to some of the new trees
        4. Removes duplicate trees from the population
        """
        logger.info("Starting evolution iteration")
        if self.fitnesses is None:
            self.fitnesses = self._calculate_fitnesses(self.population).round(
                3
            )  # this generally unnecessarily happens again > probably not with the if

        logger.debug("Performing tournament selection and crossover")
        assert self.fitnesses.shape[0] == len(self.population)
        crossover_count = self._perform_crossovers(self.fitnesses)
        assert self.fitnesses.shape[0] == len(self.population)
        logger.debug(f"Performed {crossover_count} crossover operations")

        logger.debug("Applying mutations")
        mutation_count = self._mutate_additional_population()
        assert self.fitnesses.shape[0] == len(self.population)
        logger.info(f"Applied {mutation_count} mutations")

        joined_population = np.array(self.population + self.additional_population)  # maybe worth it to calculated fitnesses first?
        codes = np.array([tree.__repr__() for tree in joined_population])
        mask = first_uniques_mask(codes)
        self.population = list(joined_population[mask])
        self.fitnesses = self._calculate_fitnesses(self.population).round(3)
        assert self.fitnesses.shape[0] == len(self.population)

        logger.debug(f"Removed {len(joined_population) - sum(mask)} duplicate trees")
        logger.debug(f"New population size: {len(self.population)}")

        self.population, self.fitnesses = choose_pareto_then_proximity(
            self.population, self.fitnesses, self.objectives, self.population_size, self.minimize_node_count
        )

        assert self.fitnesses.shape[0] == len(self.population)

        self.additional_population = []

    def _within_caps(self, tree: Tree) -> bool:
        """Return whether ``tree`` respects the configured ``max_depth`` / ``max_nodes`` caps.

        With both caps at their ``None`` default this is always ``True`` (uncapped). When a
        cap is set, variation now selects only cap-legal moves up front (deterministic
        legal-target append + legal-pair crossover), so this is used as a cheap *invariant
        assertion* after each operator rather than a generate-and-reject gate.
        """
        if self.max_nodes is not None and tree.nodes_count > self.max_nodes:
            return False
        if self.max_depth is not None and tree.depth > self.max_depth:
            return False
        return True

    def _perform_crossovers(self, fitnesses: npt.NDArray[np.float64]):
        crossover_count = 0
        target = self.population_multiplier * self.population_size
        while len(self.additional_population) < target:
            idx1, idx2 = tournament_selection_indexes(fitnesses, self.tournament_size, self.optimal_point)
            parent_1, parent_2 = self.population[idx1], self.population[idx2]
            # Caps are honoured by construction: crossover draws only from points whose
            # offspring stay within budget (value root x root is always legal, so a usable
            # pair always exists). No retry/fallback needed; assert the invariant.
            new_tree_1, new_tree_2 = crossover(parent_1, parent_2, max_depth=self.max_depth, max_nodes=self.max_nodes)
            assert self._within_caps(new_tree_1) and self._within_caps(new_tree_2)
            self.additional_population += [new_tree_1, new_tree_2]
            crossover_count += 1
        return crossover_count

    def _mutate_additional_population(self) -> int:
        mutation_count = 0
        trees_to_add = []

        for tree in self.additional_population:
            current_tree = tree

            # Structural mutation
            if np.random.rand() < tree.mutation_chance:
                allowed_mutations = get_allowed_mutations(tree, max_depth=self.max_depth, max_nodes=self.max_nodes)
                if allowed_mutations:
                    chosen_mutation = np.random.choice(np.array(allowed_mutations))
                    logger.trace(f"Applying structural mutation: {chosen_mutation.__name__}")
                    mutated = chosen_mutation(
                        tree,
                        models=self.models,
                        ids=self.ids,
                        allowed_ops=self.allowed_ops,
                        max_depth=self.max_depth,
                        max_nodes=self.max_nodes,
                    )
                    # append draws only cap-legal targets; the shrinking mutations cannot
                    # exceed a cap a within-caps parent already met. So the result is always
                    # within caps -> assert the invariant instead of reverting.
                    assert self._within_caps(mutated)
                    current_tree = mutated
                    mutation_count += 1

            # Parameter mutation (applied to structurally mutated tree if any, else original)
            if np.random.rand() < tree.mutation_chance:
                logger.trace("Applying parameter mutation")
                current_tree = mutate_parameters(current_tree, mutation_strength=self.mutation_strength)
                mutation_count += 1

            # Only add if any mutation occurred
            if current_tree is not tree:
                trees_to_add.append(current_tree)

        self.additional_population.extend(trees_to_add)
        return mutation_count

    def train(self, iterations: int):
        """
        Run the evolutionary algorithm for a specified number of iterations.

        Args:
            iterations: Number of evolution iterations to run
        """
        logger.info(f"Starting evolution with {iterations} iterations")
        self._call_hook("on_evolution_start")

        for i in tqdm.tqdm(range(iterations)):
            logger.info(f"Generation {i + 1}/{iterations}")
            self._call_hook("on_generation_start")  # possibly move to run_iteration instead
            self.run_iteration()
            self._call_hook("on_generation_end")

            if self.should_stop:
                logger.info("Early stopping triggered")
                break

        logger.info("Evolution complete")
        self._call_hook("on_evolution_end")

    def _build_train_tensors(self, preds_source, gt_path):
        """
        Load prediction tensors and ground truth from files.

        Args:
            preds_source: Source of model predictions (path or iterable of paths)
            gt_path: Path to ground truth data

        Returns:
            Tuple of (train_tensors dictionary, ground truth tensor)
        """
        logger.info("Loading prediction tensors and ground truth")
        tensor_paths = []
        if isinstance(preds_source, str):
            preds_source = Path(preds_source)
        if isinstance(preds_source, Path):
            logger.debug(f"Scanning directory for tensors: {preds_source}")
            tensor_paths = list(preds_source.glob("*"))
        elif hasattr(preds_source, "__iter__"):
            marked_paths, all_same = mark_paths(preds_source)
            if all_same:
                if marked_paths[0] == "dir":
                    for pred_source in preds_source:
                        pred_source = Path(pred_source)
                        tensor_paths += list(pred_source.glob("*"))
                elif marked_paths[0] == "file":
                    tensor_paths = list(preds_source)
            else:
                raise ValueError(
                    "preds source must be either path to directory with predictions,"
                    " list of paths to directories with predictions, or list of paths to predictions"
                )

        train_tensors = {}
        for tensor_path in tensor_paths:
            logger.debug(f"Loading tensor: {tensor_path}")
            tensor_id = Path(tensor_path).name
            if tensor_id not in train_tensors:
                train_tensors[tensor_id] = B.load(tensor_path, DEVICE)
            else:
                train_tensors[tensor_id] = B.concat([train_tensors[tensor_id], B.load(tensor_path, DEVICE)])

        logger.debug(f"Loaded {len(train_tensors)} prediction tensors")
        logger.debug(f"Loading ground truth from: {gt_path}")

        gt_tensor: None | Tensor = None
        if isinstance(gt_path, str):
            gt_path = Path(gt_path)
        if isinstance(gt_path, Path):
            if os.path.isdir(gt_path):
                for path in gt_path.glob("*"):
                    if gt_tensor is None:
                        gt_tensor = B.load(path)
                    else:
                        gt_tensor = B.concat([gt_tensor, B.load(path, device=DEVICE)])  # type: ignore
            else:
                gt_tensor = B.load(gt_path)
        elif hasattr(gt_path, "__iter__"):
            for path in gt_path:
                if gt_tensor is None:
                    gt_tensor = B.load(path)
                else:
                    gt_tensor = B.concat([gt_tensor, B.load(path, device=DEVICE)])  # type: ignore
        else:
            raise ValueError(f"{gt_path} is not valid for loading gt")

        logger.info("Tensors loaded successfully")
        return train_tensors, gt_tensor

    def _validate_input(self, fix_swapped=True):  # no way to change this argument for now TODO
        """
        Validate that all input tensors have compatible shapes.

        Checks if all prediction tensors have the same shape and if the ground truth
        tensor has a compatible shape. Can optionally fix swapped dimensions in the
        ground truth tensor.

        Args:
            fix_swapped: If True, attempts to fix swapped dimensions in ground truth tensor

        Raises:
            ValueError: If tensor shapes are incompatible and cannot be fixed
        """
        logger.info("Validating input tensors")
        # check if all tensors have the same shape
        shapes = [B.shape(tensor) for tensor in self.train_tensors.values()]

        if len(set(shapes)) > 1:
            logger.error(f"Tensors have different shapes: {shapes}")
            raise ValueError(f"Tensors have different shapes: {shapes}")

        logger.debug(f"All prediction tensors have shape: {shapes[0]}")
        logger.debug(f"Ground truth tensor has shape: {B.shape(self.gt_tensor)}")

        if B.shape(self.gt_tensor) != shapes[0]:
            gt_shape = B.shape(self.gt_tensor)
            if len(shapes[0]) > 1 and (len(gt_shape) == 1 or gt_shape[-1] == 1):
                pass
            elif fix_swapped:
                if (shapes[0] == B.shape(self.gt_tensor)[::-1]) and (len(shapes[0]) == 2):
                    logger.warning(f"Ground truth tensor dimensions appear to be swapped. Reshaping from {B.shape(self.gt_tensor)} to {shapes[0]}")
                    self.gt_tensor = B.reshape(self.gt_tensor, shapes[0])
                    logger.info("Tensor shapes fixed successfully")
                else:
                    logger.error(f"Ground truth tensor shape {B.shape(self.gt_tensor)} incompatible with prediction tensor shape {shapes[0]}")
                    raise ValueError(f"Ground truth tensor has incompatible shape: {B.shape(self.gt_tensor)} vs {shapes[0]}")
            else:
                logger.error(f"Ground truth tensor shape {B.shape(self.gt_tensor)} does not match prediction tensor shape {shapes[0]}")
                raise ValueError(f"Ground truth tensor has different shape than input tensors: {shapes[0]} != {B.shape(self.gt_tensor)}")

        logger.info("Input validation successful")

    @property
    def pareto_trees(self) -> List[Tree]:
        assert isinstance(self.fitnesses, np.ndarray), "Fitnesses not yet initialized. Did you run any iteration?"
        all_pareto_trees, _ = choose_pareto(self.population, self.fitnesses, len(self.population), self.objectives, self.minimize_node_count)
        return all_pareto_trees

    @property
    def pareto_fitnesses(self) -> np.ndarray:
        assert isinstance(self.fitnesses, np.ndarray), "Fitnesses not yet initialized. Did you run any iteration?"
        _, pareto_fitnesses = choose_pareto(self.population, self.fitnesses, len(self.population), self.objectives, self.minimize_node_count)
        return pareto_fitnesses
