import numpy as np

from app.services import embedding_cache


def setup_function(_):
    embedding_cache.clear()


def test_get_returns_none_on_miss():
    assert embedding_cache.get("never seen") is None


def test_put_then_get():
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding_cache.put("hello", vec)
    cached = embedding_cache.get("hello")
    assert cached is not None
    assert np.array_equal(cached, vec)


def test_normalization_makes_keys_equivalent():
    vec = np.array([1.0], dtype=np.float32)
    embedding_cache.put("Hello World", vec)
    assert embedding_cache.get("hello world") is not None
    assert embedding_cache.get("  HELLO WORLD  ") is not None


def test_clear_removes_all():
    embedding_cache.put("a", np.array([1.0]))
    embedding_cache.put("b", np.array([2.0]))
    embedding_cache.clear()
    assert embedding_cache.get("a") is None
    assert embedding_cache.get("b") is None
