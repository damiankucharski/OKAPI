from okapi.globals import BACKEND as B
from okapi.globals import set_postprocessing_function
from okapi.lib_types import Tensor


def scale_vector_to_sum_1(tensor: Tensor):
    """
    Normalize a tensor so that each vector sums to 1.

    This function scales each vector in the tensor along the last dimension
    such that its elements sum to 1, making it suitable for representing
    probability distributions.

    Args:
        tensor: Input tensor to be normalized

    Returns:
        Normalized tensor where each vector sums to 1
    """
    mass = B.unsqueeze(B.sum(tensor, axis=-1), -1)
    # A minimum over disjoint expert supports can remove all class mass.
    # Unit pseudocounts give just those rows a uniform distribution.
    empty = mass == 0
    counts = tensor + empty
    return counts / B.unsqueeze(B.sum(counts, axis=-1), -1)


def set_multiclass_postprocessing():
    """
    Configure the global postprocessing function for multiclass classification.

    This function sets the global postprocessing to normalize tensor outputs so that
    they can be interpreted as probability distributions, which is required for
    multiclass classification tasks. This is important if probas and not logits are an output.
    """
    set_postprocessing_function(scale_vector_to_sum_1)
