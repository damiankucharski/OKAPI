# Backend API

This section documents the backend components of OKAPI, which handle tensor operations.

## Backend Interface

The abstract interface that all backend implementations must follow.

```python
from okapi.backend.backend_interface import BackendInterface
```

::: okapi.backend.backend_interface.BackendInterface
    options:
      show_source: true

## Backend Factory

Factory class for managing tensor backends.

```python
from okapi.backend.backend import Backend
```

::: okapi.backend.backend.Backend
    options:
      show_source: true

## Global Backend Configuration

Functions and variables for configuring the backend.

```python
from okapi.globals import BACKEND, set_backend, get_backend, DEVICE
```

::: okapi.globals
    options:
      members:
        - BACKEND
        - set_backend
        - get_backend
        - DEVICE
      show_source: true
