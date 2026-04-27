"""Deterministic query understanding (no LLM).

자연어 쿼리에서 lang/recency/카테고리 힌트와 정제된 search_keywords를 추출한다.
이전에는 Qwen2.5-0.5B GGUF를 llama.cpp로 호출했으나, 출력의 거의 모든 신호가
deterministic regex(`_sanitize`)에 의해 덮어씌워지는 구조였고 LLM 자체의
hallucination이 정확도를 오히려 떨어뜨렸다. 모델·llama-cpp-python 의존을 제거하고
regex만 남긴다. API contract(`POST /search/understand`)는 그대로 유지되며 응답
schema 호환.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

_HAS_KOREAN_RE = re.compile(r"[가-힣]")
_LANG_KO_HINT_RE = re.compile(r"한국어|한글|korean|korean글", re.IGNORECASE)
_LANG_EN_HINT_RE = re.compile(r"english|영어로|영문|in english", re.IGNORECASE)
_RECENCY_DAY_RE = re.compile(r"오늘|어제|today|yesterday|최근\s*하루")
_RECENCY_WEEK_RE = re.compile(
    r"지난\s*주|이번\s*주|최근\s*주|최근\s*일주일|일주일|last\s*week|this\s*week|past\s*week",
    re.IGNORECASE,
)
_RECENCY_MONTH_RE = re.compile(
    r"지난\s*(?:한\s*)?달|이번\s*달|지난\s*개월|최근\s*(?:한\s*)?달|최근\s*개월|"
    r"이번\s*\d+\s*달|한\s*달간|last\s*month|this\s*month|past\s*month",
    re.IGNORECASE,
)
_RECENCY_YEAR_RE = re.compile(
    r"올해|작년|이번\s*년|last\s*year|this\s*year|past\s*year|최근\s*\d+\s*년",
    re.IGNORECASE,
)
_TIME_WORDS_RE = re.compile(
    r"오늘의?|어제의?|"
    r"지난\s*(?:한\s*)?(?:주|달|개월|년|일주일|하루)(?:간|동안)?|"
    r"이번\s*(?:주|달|년|일주일|하루)(?:간|동안)?|"
    r"최근\s*(?:한\s*)?(?:주|달|개월|년|일주일|하루)(?:간|동안)?|"
    r"올해\s*(?:나온|올라온|공개된|작성된)|"
    r"한\s*달간|일주일(?:간)?",
    re.IGNORECASE,
)
_LANG_WORDS_RE = re.compile(
    r"한국어|한글|영어로?|영문|in\s*english|english\s*only|korean", re.IGNORECASE
)
# 검색에 무의미한 한국어 의문/요청 어미 — keywords에서 제거
_INTENT_NOISE_RE = re.compile(
    r"\b(?:찾아\s*줘|찾아|찾기|보여\s*줘|알려\s*줘|관련\s*글|관련된|글|기사|article)s?\b",
    re.IGNORECASE,
)

_VALID_CATEGORY_HINTS = {
    "AI/ML", "Backend", "Frontend", "DevOps", "Mobile", "Data", "Infra"
}


def analyze(query: str) -> dict[str, Any]:
    """Deterministic 추출. LLM 호출 없음.

    Returns:
        {
          "search_keywords": str | None,
          "lang": "ko" | "en" | None,
          "recency": "day" | "week" | "month" | "year" | None,
          "category_hint": None,  # heuristic 미정확 → 항상 None (후속 작업에서 BERT-tiny intent classifier 검토)
          "intent": str | None,
        }
    """
    parsed: dict[str, Any] = {
        "search_keywords": None,
        "lang": None,
        "recency": None,
        "category_hint": None,
        "intent": None,
    }
    if not query or not query.strip():
        return parsed

    # lang
    if _LANG_KO_HINT_RE.search(query):
        parsed["lang"] = "ko"
    elif _LANG_EN_HINT_RE.search(query):
        parsed["lang"] = "en"

    # recency
    for rx, label in (
        (_RECENCY_DAY_RE, "day"),
        (_RECENCY_WEEK_RE, "week"),
        (_RECENCY_MONTH_RE, "month"),
        (_RECENCY_YEAR_RE, "year"),
    ):
        if rx.search(query):
            parsed["recency"] = label
            break

    # search_keywords: 원문에서 시간/언어/의문 어미 단어 제거
    kw = query.strip()
    kw = _TIME_WORDS_RE.sub(" ", kw)
    kw = _LANG_WORDS_RE.sub(" ", kw)
    kw = _INTENT_NOISE_RE.sub(" ", kw)
    kw = re.sub(r"\s+", " ", kw).strip(" .,?!。")
    parsed["search_keywords"] = kw or None

    # intent: 정제된 keywords가 곧 의도. 별도 LLM 추론 없음.
    parsed["intent"] = parsed["search_keywords"]
    return parsed


def recency_to_published_after(recency: Optional[str]) -> Optional[float]:
    if not recency:
        return None
    now = datetime.now(timezone.utc)
    delta = {
        "day": timedelta(days=1),
        "week": timedelta(days=7),
        "month": timedelta(days=30),
        "year": timedelta(days=365),
    }.get(recency)
    if delta is None:
        return None
    return (now - delta).timestamp()
