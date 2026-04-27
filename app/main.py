import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import api
from app.db.rds_conn import engine
from app.middleware.request_logger import install as install_request_logger

load_dotenv()

log = logging.getLogger(__name__)


async def _drift_loop() -> None:
    """매 24h 인코더 parity 측정. SentenceTransformer 미설치 등으로 실패해도 loop 유지."""
    from app.core.config import config as _cfg
    from app.services import drift

    if not _cfg.DRIFT_ENABLED:
        return
    try:
        await asyncio.sleep(_cfg.DRIFT_INITIAL_DELAY_SEC)
    except asyncio.CancelledError:
        return
    while True:
        try:
            await asyncio.to_thread(drift.measure)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.warning("drift loop iteration failed: %s", exc)
        try:
            await asyncio.sleep(_cfg.DRIFT_INTERVAL_SEC)
        except asyncio.CancelledError:
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.config import config as _cfg
    from app.services.embedder import QueryEncoder
    from app.services.qdrant_client import ensure_payload_indexes

    log.info("Starting up: loading query encoder...")
    encoder = QueryEncoder.instance()
    encoder.warmup()
    log.info("Query encoder ready.")

    if _cfg.RERANKER_ENABLED:
        try:
            from app.services.reranker import Reranker
            log.info("Loading cross-encoder reranker...")
            reranker = Reranker.instance()
            reranker.warmup()
            log.info("Reranker ready.")
        except FileNotFoundError as exc:
            log.warning("Reranker model not found, hybrid_rerank mode will fall back: %s", exc)
        except Exception as exc:
            log.warning("Reranker init failed, hybrid_rerank mode will fall back: %s", exc)

    try:
        ensure_payload_indexes()
    except Exception as exc:
        log.warning("Qdrant payload index bootstrap failed: %s", exc)

    drift_task: asyncio.Task | None = None
    if _cfg.DRIFT_ENABLED:
        drift_task = asyncio.create_task(_drift_loop(), name="drift-loop")
        log.info(
            "Drift loop scheduled (initial=%ds, interval=%ds).",
            _cfg.DRIFT_INITIAL_DELAY_SEC, _cfg.DRIFT_INTERVAL_SEC,
        )

    log.info("DB engine ready.")
    yield
    log.info("Shutting down...")
    if drift_task is not None:
        drift_task.cancel()
    if engine is not None:
        engine.dispose()


app = FastAPI(lifespan=lifespan)
install_request_logger(app)
app.include_router(api)
