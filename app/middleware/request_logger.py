"""metric.bite-sized.xyz 대시보드용 온라인 요청 로거.

매 /feeds·/search 요청에 대해 latency, status, query 등을 비동기로
recsys_request_log 테이블에 적재한다. 응답 경로를 막지 않기 위해
asyncio.to_thread로 sync DB write를 thread로 분리하고, DB 적재 실패는
조용히 삼킨다 (테이블 부재 등에 graceful skip).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request, Response

from app.db.rds_conn import get_sessionmaker
from app.models.request_log import RecsysRequestLog

logger = logging.getLogger(__name__)


def _classify_endpoint(path: str) -> Optional[str]:
    if path.startswith("/feeds"):
        return "feeds"
    if path.startswith("/search"):
        return "search"
    return None


def _extract_member_id(request: Request) -> Optional[int]:
    raw = request.query_params.get("member_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _persist_sync(
    *,
    request_id: str,
    endpoint: str,
    member_id: Optional[int],
    status_code: int,
    latency_ms: int,
    query_text: Optional[str],
    error_class: Optional[str],
) -> None:
    try:
        SessionLocal = get_sessionmaker()
        with SessionLocal() as session:
            session.add(
                RecsysRequestLog(
                    request_id=request_id,
                    endpoint=endpoint,
                    member_id=member_id,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    query_text=(query_text[:255] if query_text else None),
                    error_class=error_class,
                )
            )
            session.commit()
    except Exception:  # noqa: BLE001 — 테이블 부재 / 일시 장애 등 모두 graceful skip
        logger.debug("request_logger persist skipped", exc_info=True)


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logger(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        endpoint = _classify_endpoint(request.url.path)
        if endpoint is None:
            return await call_next(request)

        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        query_text = request.query_params.get("query")
        member_id = _extract_member_id(request)

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            asyncio.create_task(
                asyncio.to_thread(
                    _persist_sync,
                    request_id=request_id,
                    endpoint=endpoint,
                    member_id=member_id,
                    status_code=500,
                    latency_ms=latency_ms,
                    query_text=query_text,
                    error_class=type(exc).__name__,
                )
            )
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        asyncio.create_task(
            asyncio.to_thread(
                _persist_sync,
                request_id=request_id,
                endpoint=endpoint,
                member_id=member_id,
                status_code=response.status_code,
                latency_ms=latency_ms,
                query_text=query_text,
                error_class=None,
            )
        )
        response.headers["X-Request-Id"] = request_id
        return response
