import gc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from okapi.okapi import Okapi


class Callback:
    """
    Base class for callbacks that can be triggered during the evolutionary process.

    Callbacks allow monitoring and potentially modifying the evolution process at
    specific points: before/after each generation, and at the start/end of the
    entire evolution. Custom callbacks should inherit from this class and override
    the methods corresponding to the desired intervention points.
    """

    def __init__(self) -> None:
        pass

    def on_generation_end(self, okapi: "Okapi") -> None:
        """
        Called at the end of each generation.

        This hook is triggered after a generation (iteration) of evolution has completed,
        including selection, crossover, and mutation operations.

        Args:
            okapi: The Okapi instance running the evolution
        """
        pass

    def on_evolution_end(self, okapi: "Okapi") -> None:
        """
        Called at the end of the entire evolution process.

        This hook is triggered when all generations have been completed or
        when the evolution process is manually stopped.

        Args:
            okapi: The Okapi instance running the evolution
        """
        pass

    def on_evolution_start(self, okapi: "Okapi") -> None:
        """
        Called at the start of the evolution process.

        This hook is triggered before any generations are run, after the
        initial population has been created.

        Args:
            okapi: The Okapi instance running the evolution
        """
        pass

    def on_generation_start(self, okapi: "Okapi") -> None:
        """
        Called at the start of each generation.

        This hook is triggered before a generation (iteration) of evolution begins,
        before any selection, crossover, or mutation operations.

        Args:
            okapi: The Okapi instance running the evolution
        """
        pass


class FitnessNoChangeEarlyStoppingCallback(Callback):
    def __init__(self, n_iterations=5):
        super().__init__()
        self._iterations_no_change = 0
        self._last_fitnesses = None
        self._n_iterations = n_iterations

    def on_generation_end(self, okapi: "Okapi") -> None:
        if self._last_fitnesses is None:
            self._last_fitnesses = okapi.fitnesses
            return
        assert isinstance(okapi.fitnesses, np.ndarray), "Fitnesses need to be numpy array"
        if np.allclose(self._last_fitnesses, okapi.fitnesses):
            self._iterations_no_change += 1
            if self._iterations_no_change >= self._n_iterations:
                okapi.should_stop = True
        else:
            self._iterations_no_change = 0


class MemoryCleanupCallback(Callback):
    """
    Callback that triggers garbage collection at specified intervals.

    Useful for large tensor operations where Python's automatic garbage
    collection may not be aggressive enough.

    Args:
        gc_every_n_generations: Run garbage collection every N generations.
                               Default is 1 (every generation).
        clear_cuda_cache: If True, also clears CUDA cache when available.
                         Default is True.
    """

    def __init__(self, gc_every_n_generations: int = 1, clear_cuda_cache: bool = True):
        super().__init__()
        self._gc_interval = gc_every_n_generations
        self._clear_cuda_cache = clear_cuda_cache
        self._generation_count = 0

    def on_generation_end(self, okapi: "Okapi") -> None:
        self._generation_count += 1
        if self._generation_count % self._gc_interval == 0:
            # Clear evaluation caches from all trees in population
            for tree in okapi.population:
                tree._clean_evals()
            if hasattr(okapi, "additional_population") and okapi.additional_population:
                for tree in okapi.additional_population:
                    tree._clean_evals()

            gc.collect()
            if self._clear_cuda_cache:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
