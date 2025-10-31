# OKAPI: Genetic Programming for Ensemble Model Fusion

OKAPI is a Python library that uses genetic programming to evolve optimal ensembles of machine learning models for classification tasks. It combines the predictions of multiple models into a single, more accurate prediction by evolving tree structures representing different fusion strategies.

## Key Features

- **Tree-based representation**: Models ensemble architectures as trees with models as leaves and fusion operations as nodes
- **Evolutionary optimization**: Uses genetic programming with crossover and mutation operations to find optimal fusion strategies
- **Multiple fusion operations**: Supports mean, min, max, weighted mean, and other operations to combine predictions
- **Backend flexibility**: Works with both NumPy and PyTorch backends
- **Pareto optimization**: Balances model complexity and performance for robust solutions

## When to Use OKAPI

OKAPI is particularly useful when:

- You have multiple models predicting the same target
- You want to combine these models in a way that outperforms individual models
- You need interpretable fusion structures that show how models are combined
- You want to automatically discover which models are most useful for your task

## Project Status

OKAPI is currently under active development. The core functionality is implemented, but the library may change significantly before the first stable release.

## Documentation Structure

- **API Reference**: Detailed documentation for all classes and functions in the library
- **Development**: Information for contributors interested in helping with the development
