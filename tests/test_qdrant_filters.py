from app.services.qdrant_client import SearchFilters, to_qdrant_filter


class TestSearchFilters:
    def test_is_empty_when_all_none(self):
        assert SearchFilters().is_empty()

    def test_not_empty_when_any_set(self):
        assert not SearchFilters(category_id=1).is_empty()
        assert not SearchFilters(lang="ko").is_empty()
        assert not SearchFilters(blog_id=42).is_empty()
        assert not SearchFilters(published_after=1.0).is_empty()


class TestToQdrantFilter:
    def test_returns_none_when_empty(self):
        assert to_qdrant_filter(SearchFilters()) is None

    def test_category_filter(self):
        f = to_qdrant_filter(SearchFilters(category_id=2))
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "category_id"

    def test_combined_filters(self):
        f = to_qdrant_filter(
            SearchFilters(category_id=2, lang="ko", blog_id=42, published_after=1.0)
        )
        assert f is not None
        keys = [c.key for c in f.must]
        assert keys == ["category_id", "lang", "blog_id", "published_at"]

    def test_published_range_combined_into_one_condition(self):
        f = to_qdrant_filter(SearchFilters(published_after=1.0, published_before=2.0))
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "published_at"
        assert f.must[0].range.gte == 1.0
        assert f.must[0].range.lte == 2.0
