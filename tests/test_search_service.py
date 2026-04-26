import pytest

from app.api.search.service import _decode_cursor, _encode_cursor


class TestCursor:
    def test_round_trip(self):
        for offset in (0, 10, 50, 999):
            assert _decode_cursor(_encode_cursor(offset)) == offset

    def test_none_returns_zero(self):
        assert _decode_cursor(None) == 0

    def test_empty_returns_zero(self):
        assert _decode_cursor("") == 0

    def test_invalid_cursor_raises(self):
        with pytest.raises(ValueError):
            _decode_cursor("not-base64!!!")

    def test_offset_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _decode_cursor(_encode_cursor(10001))

    def test_negative_offset_raises(self):
        with pytest.raises(ValueError):
            _decode_cursor(_encode_cursor(-1))
