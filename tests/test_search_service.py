import pytest

from app.api.search.service import (
    _decode_cursor,
    _encode_cursor,
    _new_query_id,
    _query_hash,
)


class TestCursor:
    def test_round_trip_preserves_snapshot_offset_hash_and_query_id(self):
        snap = [f"a{i}" * 9 for i in range(50)]
        h = _query_hash("hello world")
        qid = _new_query_id()
        cur = _encode_cursor(snap, 20, h, qid)
        decoded = _decode_cursor(cur)
        assert decoded["f"] == snap
        assert decoded["o"] == 20
        assert decoded["q"] == h
        assert decoded["i"] == qid

    def test_none_returns_empty_dict(self):
        assert _decode_cursor(None) == {}

    def test_empty_returns_empty_dict(self):
        assert _decode_cursor("") == {}

    def test_invalid_cursor_raises(self):
        with pytest.raises(ValueError):
            _decode_cursor("not-base64!!!")

    def test_corrupt_payload_raises(self):
        with pytest.raises(ValueError):
            _decode_cursor("aGVsbG8=")

    def test_query_hash_stable(self):
        assert _query_hash("LLM 추천") == _query_hash("  LLM 추천  ")
        assert _query_hash("a") != _query_hash("b")

    def test_query_id_is_unique(self):
        ids = {_new_query_id() for _ in range(100)}
        assert len(ids) == 100

    def test_cursor_is_url_safe_with_query_id(self):
        snap = ["x" * 27 for _ in range(100)]
        cur = _encode_cursor(snap, 0, _query_hash("q"), _new_query_id())
        assert all(c.isalnum() or c in "-_=" for c in cur), f"non-url-safe in cursor: {cur}"
        assert len(cur) < 8000
