"""LLM-based query understanding (CPU, llama.cpp).

자연어 쿼리(예: "지난 주 한국어 LLM 추천 시스템 글 찾아")를 받아 검색 키워드 +
필터 힌트(lang, recency, category 등)를 구조화된 JSON으로 추출한다.

모델: Qwen2.5-0.5B-Instruct (Q4_0 GGUF, ~400MB). CPU 단일 호출 latency ~500ms.
검색 latency 직격을 피하기 위해 별도 엔드포인트(/search/understand)에서만 호출하고,
검색 호출 자체는 raw query 그대로 처리한다. 클라이언트가 두 호출을 조합한다.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import config

log = logging.getLogger(__name__)


_SYSTEM = """You analyze a user's tech blog search query (Korean or English) and extract structured intent.
Output a JSON object only.

Rules:
  - search_keywords: string. 핵심 명사/기술용어만. 조사·부사·"찾아줘" 등 동사 제거.
  - lang: "ko" | "en" | null. user explicitly asks for Korean/English content. Otherwise null.
  - recency: "day" | "week" | "month" | "year" | null. ONLY if user explicitly mentions time. Otherwise null.
  - category_hint: "AI/ML" | "Backend" | "Frontend" | "DevOps" | "Mobile" | "Data" | "Infra" | null. ONLY if topic obviously fits. Otherwise null.
  - intent: short ≤ 30 char description.

Examples:

Q: 지난 주 한국어 LLM 추천 시스템 글 찾아줘
A: {"search_keywords":"LLM 추천 시스템","lang":"ko","recency":"week","category_hint":"AI/ML","intent":"LLM 추천 시스템"}

Q: kubernetes operator pattern
A: {"search_keywords":"kubernetes operator pattern","lang":null,"recency":null,"category_hint":"DevOps","intent":"k8s operator"}

Q: 쿠버네티스
A: {"search_keywords":"쿠버네티스","lang":null,"recency":null,"category_hint":"DevOps","intent":"k8s"}

Q: 최근 한 달간 카카오 블로그
A: {"search_keywords":"카카오","lang":null,"recency":"month","category_hint":null,"intent":"카카오 글"}

Q: 오늘의 도커 컨테이너 사용법
A: {"search_keywords":"도커 컨테이너","lang":null,"recency":"day","category_hint":"DevOps","intent":"도커 사용법"}"""


class QueryUnderstander:
    _instance: "QueryUnderstander | None" = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "QueryUnderstander":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        from llama_cpp import Llama  # heavy import deferred until first use

        model_path = Path(config.QU_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"QU model not found: {model_path}. "
                "Run scripts/download_qu_llm.py to fetch."
            )

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=config.QU_N_CTX,
            n_threads=config.QU_N_THREADS,
            verbose=False,
        )

    def warmup(self) -> None:
        self.analyze("warmup")

    def analyze(self, query: str) -> dict[str, Any]:
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            max_tokens=config.QU_MAX_TOKENS,
            temperature=0.0,
        )
        text = out["choices"][0]["message"]["content"].strip()
        parsed = _safe_parse_json(text)
        return _sanitize(parsed, query)


_HAS_KOREAN_RE = re.compile(r"[가-힣]")
_HAS_LATIN_RE = re.compile(r"[a-zA-Z]")
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
    r"오늘의?|어제의?|지난\s*(?:한\s*)?(?:주|달|개월|년)|이번\s*(?:주|달|년)|"
    r"최근\s*(?:한\s*)?(?:주|달|개월|일주일|하루)|올해\s*나온|올해\s*올라온|올해\s*공개된|"
    r"한\s*달간|일주일",
    re.IGNORECASE,
)
_LANG_WORDS_RE = re.compile(r"한국어|한글|영어로?|영문|in\s*english|english\s*only|korean", re.IGNORECASE)


def _sanitize(parsed: dict[str, Any], query: str) -> dict[str, Any]:
    """LLM의 hallucination을 deterministic rule로 보정.

    - lang: query 텍스트에 명시적 lang 힌트가 없으면 null로 강제 (작은 모델이
      query에 한글이 있다는 이유만으로 lang=ko를 채우는 경향이 있음)
    - recency: query에 시간 키워드가 없으면 null로 강제
    - search_keywords: 영문으로 임의 번역되는 경우(원문이 한글일 때) 원문 사용
    """
    has_korean = bool(_HAS_KOREAN_RE.search(query))
    has_latin = bool(_HAS_LATIN_RE.search(query))

    if _LANG_KO_HINT_RE.search(query):
        parsed["lang"] = "ko"
    elif _LANG_EN_HINT_RE.search(query):
        parsed["lang"] = "en"
    else:
        parsed["lang"] = None

    detected: Optional[str] = None
    for re_, label in (
        (_RECENCY_DAY_RE, "day"),
        (_RECENCY_WEEK_RE, "week"),
        (_RECENCY_MONTH_RE, "month"),
        (_RECENCY_YEAR_RE, "year"),
    ):
        if re_.search(query):
            detected = label
            break
    parsed["recency"] = detected

    kw = parsed.get("search_keywords") or ""
    kw_has_korean = bool(_HAS_KOREAN_RE.search(kw))
    if has_korean and not kw_has_korean:
        # LLM이 한국어 query를 영문으로 번역해 쓴 경우 → 원문 fallback
        kw = query.strip()
    # 시간/언어 단어 제거 (별도 필터로 이미 추출했으므로)
    kw = _TIME_WORDS_RE.sub(" ", kw)
    kw = _LANG_WORDS_RE.sub(" ", kw)
    kw = re.sub(r"\s+", " ", kw).strip(" .,?!。")
    parsed["search_keywords"] = kw or None

    if parsed.get("category_hint") not in (
        "AI/ML", "Backend", "Frontend", "DevOps", "Mobile", "Data", "Infra", None
    ):
        parsed["category_hint"] = None

    return parsed


def _safe_parse_json(text: str) -> dict[str, Any]:
    # Llama가 가끔 ```json ... ``` 코드 블록을 포함시키는 경우가 있어 정제.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # JSON object substring을 best-effort로 추출
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


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
