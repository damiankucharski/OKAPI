"""
GIRAFFE: Genetic Programming for Ensemble Model Fusion

GIRAFFE is a Python library that uses genetic programming to evolve optimal
ensembles of machine learning models for classification tasks. It combines
the predictions of multiple models into a single, more accurate prediction
by evolving tree structures representing different fusion strategies.

Key components:
- Tree-based representation of ensemble models
- Evolution through crossover and mutation operations
- Support for different fusion operations (mean, min, max, weighted mean)
- Multiple backend options (NumPy, PyTorch)
- Pareto optimization for balancing model complexity and performance
"""

import os
import sys

from loguru import logger

# Configure loguru logger
log_level = os.environ.get("GIRAFFE_LOG_LEVEL", "INFO")
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=log_level)


from giraffe.giraffe import Giraffe  # noqa E402

logger.trace(f"{Giraffe}")
