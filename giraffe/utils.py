import os
import pickle

import numpy as np
from loguru import logger


class Pickle:
    """
    A utility class for serializing and deserializing Python objects using pickle.

    This class provides static methods for saving objects to files and loading them back,
    which is particularly useful for persisting tree architectures.
    """

    @staticmethod
    def load(path):
        """
        Load a Python object from a pickle file.

        Args:
            path: File path to load the object from

        Returns:
            The deserialized Python object
        """
        logger.debug(f"Loading pickle file from: {path}")
        try:
            with open(path, "rb") as file:
                obj = pickle.load(file)
            logger.debug(f"Successfully loaded object from {path}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load pickle file from {path}: {str(e)}")
            raise

    @staticmethod
    def save(path, obj):
        """
        Save a Python object to a pickle file.

        Args:
            path: File path where the object will be saved
            obj: The Python object to serialize and save
        """
        logger.debug(f"Saving object to pickle file: {path}")
        try:
            with open(path, "wb") as file:
                pickle.dump(obj, file)
            logger.debug(f"Successfully saved object to {path}")
        except Exception as e:
            logger.error(f"Failed to save object to {path}: {str(e)}")
            raise


def first_uniques_mask(arr):
    """
    Create a boolean mask that identifies the first occurrence of each unique item in an array.

    This function is useful for filtering duplicates from an array while preserving the order
    of first appearances.

    Args:
        arr: An array-like object to analyze

    Returns:
        A list of booleans where True indicates the first occurrence of a value and
        False indicates a duplicate of a previously seen value
    """
    logger.trace(f"Creating unique items mask for array of length {len(arr)}")
    mask = []
    unique_count = 0

    for index, item in enumerate(arr):
        if item not in arr[:index]:
            mask.append(True)
            unique_count += 1
        else:
            mask.append(False)

    logger.trace(f"Found {unique_count} unique items out of {len(arr)} total items")
    return mask


def mark_paths(list_of_paths) -> tuple[list[str | None], bool]:
    """
    Mark each path in the list with its type (directory or file) or None if it doesn't exist.

    Args:
        list_of_paths: A list of paths to be marked.

    Returns:
        A tuple containing:
            - A list of strings or None values representing the type of each path.
            - A boolean indicating whether all paths are of the same type.
    """
    marked_paths: list[str | None] = []

    for path in list_of_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                marked_paths.append("dir")
            else:
                marked_paths.append("file")
        else:
            marked_paths.append(None)
    all_same = all(item == marked_paths[0] for item in marked_paths)
    return marked_paths, all_same


def _euclidean_distances(points, point):
    point = point.reshape(1, -1)
    squared_diff = (points - point) ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=1))

    return distances
