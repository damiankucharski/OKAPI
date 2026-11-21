import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import postprocessing_function, set_postprocessing_function, use_postprocessing_function
from okapi.node import MinNode, ValueNode
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def reset_default(monkeypatch):
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)


def test_nested_exception_and_default_updates_have_distinct_lifetimes():
    with use_postprocessing_function(lambda x: x + 1):
        assert postprocessing_function(5) == 6
        with pytest.raises(RuntimeError):
            with use_postprocessing_function(lambda x: x + 2):
                set_postprocessing_function(lambda x: x + 3)
                assert postprocessing_function(5) == 7
                raise RuntimeError("end inner scope")
        assert postprocessing_function(5) == 6
    assert postprocessing_function(5) == 8
    with pytest.raises(TypeError, match="callable"):
        with use_postprocessing_function(None):
            pass
    assert postprocessing_function(5) == 8


def test_parallel_tree_predictions_use_their_own_normalization_and_leave_default_alone():
    barrier = Barrier(3)

    def predict(scope):
        # Disjoint minimum first creates zero total mass in every class.
        tree = Tree(ValueNode([MinNode([ValueNode(None, np.array([[0., 1.]]), 1)])], np.array([[1., 0.]]), 0))
        with use_postprocessing_function(scope):
            barrier.wait(timeout=5)
            result = tree.predict()
            barrier.wait(timeout=5)
        return result, postprocessing_function(2)

    with ThreadPoolExecutor(2) as pool:
        normal = pool.submit(predict, scale_vector_to_sum_1)
        identity = pool.submit(predict, lambda x: x)
        barrier.wait(timeout=5)
        assert postprocessing_function(2) == 2
        barrier.wait(timeout=5)
        p, after_normal = normal.result()
        q, after_identity = identity.result()
    np.testing.assert_array_equal(p, [[.5, .5]])
    np.testing.assert_array_equal(q, [[0., 0.]])
    assert after_normal == after_identity == 2


def test_async_siblings_keep_separate_overrides_on_the_same_thread():
    async def observe():
        one, two = asyncio.Event(), asyncio.Event()

        async def first():
            with use_postprocessing_function(lambda x: x + 1):
                one.set()
                await two.wait()
                return postprocessing_function(0)

        async def second():
            await one.wait()
            with use_postprocessing_function(lambda x: x + 2):
                two.set()
                await asyncio.sleep(0)
                return postprocessing_function(0)

        return await asyncio.gather(first(), second())

    assert asyncio.run(observe()) == [1, 2]
    assert postprocessing_function(0) == 0
