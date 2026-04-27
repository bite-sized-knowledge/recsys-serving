from app.services.query_understanding import analyze, recency_to_published_after


class TestAnalyze:
    def test_empty_query(self):
        assert analyze("") == {
            "search_keywords": None,
            "lang": None,
            "recency": None,
            "category_hint": None,
            "intent": None,
        }

    def test_keywords_only(self):
        out = analyze("쿠버네티스")
        assert out["search_keywords"] == "쿠버네티스"
        assert out["lang"] is None
        assert out["recency"] is None

    def test_lang_korean_hint(self):
        assert analyze("한국어 LLM 글")["lang"] == "ko"
        assert analyze("Korean tech blog")["lang"] == "ko"

    def test_lang_english_hint(self):
        assert analyze("kubernetes operator pattern English")["lang"] == "en"
        assert analyze("영어로 된 React 글")["lang"] == "en"

    def test_lang_no_hint_is_null(self):
        # 한글이 query에 있다는 이유만으로 lang=ko로 설정하면 안 됨
        assert analyze("쿠버네티스 운영")["lang"] is None

    def test_recency_day(self):
        for q in ("오늘의 도커", "어제 올라온", "today's stack"):
            assert analyze(q)["recency"] == "day"

    def test_recency_week(self):
        for q in ("지난 주", "이번 주 글", "last week", "지난 일주일간"):
            assert analyze(q)["recency"] == "week"

    def test_recency_month(self):
        for q in ("지난 달", "최근 한 달간", "last month"):
            assert analyze(q)["recency"] == "month"

    def test_recency_year(self):
        for q in ("올해 나온", "작년", "this year"):
            assert analyze(q)["recency"] == "year"

    def test_keywords_strip_time_words(self):
        out = analyze("최근 한 달간 카카오 블로그")
        assert out["search_keywords"] == "카카오 블로그"
        assert out["recency"] == "month"

    def test_keywords_strip_intent_noise(self):
        out = analyze("LLM 추천 시스템 글 찾아줘")
        assert out["search_keywords"] == "LLM 추천 시스템"

    def test_category_hint_always_null(self):
        # heuristic 정확도 부족으로 항상 null. intent classifier 도입 시 변경.
        for q in ("kubernetes", "리액트 hook", "데이터 파이프라인"):
            assert analyze(q)["category_hint"] is None


class TestRecencyToPublishedAfter:
    def test_none(self):
        assert recency_to_published_after(None) is None

    def test_unknown(self):
        assert recency_to_published_after("decade") is None

    def test_day_returns_recent_epoch(self):
        import time
        ts = recency_to_published_after("day")
        assert ts is not None
        assert abs(time.time() - 86400 - ts) < 60
