from typing import Any, TypeAlias

"""
Tensor is a generic type alias that can represent any tensor-like object.
This allows the library to work with different backend implementations (NumPy, PyTorch, etc.)
"""
Tensor: TypeAlias = Any
