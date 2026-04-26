from app.services.hybrid_search import _sanitize_phrase, rrf_fuse


class TestSanitizePhrase:
    def test_strips_boolean_meta_chars(self):
        assert _sanitize_phrase('LLM "추천" +시스템') == '"LLM  추천   시스템"'

    def test_removes_minus_star_tilde(self):
        assert _sanitize_phrase("a -b *c ~d") == '"a  b  c  d"'

    def test_empty_after_sanitize(self):
        assert _sanitize_phrase('"+-*~') == ""

    def test_blank_query(self):
        assert _sanitize_phrase("   ") == ""


class TestRRFFuse:
    def test_deterministic_for_same_input(self):
        a = ["x", "y", "z"]
        b = ["y", "z", "x"]
        assert rrf_fuse(a, b) == rrf_fuse(a, b)

    def test_intersection_ranks_higher(self):
        bm25 = ["a", "b", "c", "d"]
        dense = ["c", "a", "e", "f"]
        # 'a'와 'c'는 양쪽 모두에 있고 상위에 위치 → 1, 2위
        result = rrf_fuse(bm25, dense)
        assert result[:2] == ["a", "c"]

    def test_single_ranking(self):
        assert rrf_fuse(["a", "b", "c"]) == ["a", "b", "c"]

    def test_no_duplicates(self):
        result = rrf_fuse(["a", "b", "a"], ["a", "c"])
        assert result.count("a") == 1

    def test_k_affects_score_curve(self):
        # k가 작을수록 상위 rank의 가중치가 커진다
        bm25 = ["a", "b"]
        dense = ["b", "a"]
        # k=60: 두 항목 점수 매우 비슷 → "a"가 살짝 우위 (bm25 1위 + dense 2위)
        result_default = rrf_fuse(bm25, dense, k=60)
        assert result_default[0] == "a"
