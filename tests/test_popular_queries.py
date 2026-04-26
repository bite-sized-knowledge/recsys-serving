from app.services import popular_queries, redis_client


def setup_function(_):
    popular_queries.clear()


def teardown_module(_):
    redis_client.reset()


def test_record_increments_count():
    popular_queries.record("kubernetes")
    popular_queries.record("kubernetes")
    popular_queries.record("docker")
    top = popular_queries.top(10)
    assert top[0] == "kubernetes"
    assert "docker" in top


def test_normalize_lowercases_and_trims():
    popular_queries.record("  Kubernetes  ")
    popular_queries.record("kubernetes")
    top = popular_queries.top(10)
    # 두 입력이 같은 정규화 키로 누적
    assert top.count("kubernetes") == 1


def test_suggest_prefix_match():
    popular_queries.record("kubernetes")
    popular_queries.record("kafka")
    popular_queries.record("docker")
    suggestions = popular_queries.suggest("k", limit=10)
    assert "kubernetes" in suggestions
    assert "kafka" in suggestions
    assert "docker" not in suggestions


def test_suggest_empty_prefix_returns_top():
    popular_queries.record("a")
    popular_queries.record("b")
    popular_queries.record("a")
    assert popular_queries.suggest("", limit=10)[0] == "a"


def test_too_long_query_ignored():
    popular_queries.record("a" * 300)
    assert popular_queries.top(10) == []
