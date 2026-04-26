import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import api
from app.db.rds_conn import engine
from app.middleware.request_logger import install as install_request_logger

load_dotenv()

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델/외부 인덱스 초기화는 import 시점이 아니라 lifespan에서 처리한다.
    # 검색 의존 모듈 import도 startup 시점으로 미뤄 ONNX 파일이 없는
    # 환경(테스트, 마이그레이션 등)에서 모듈 로드가 깨지지 않게 한다.
    from app.services.embedder import QueryEncoder
    from app.services.qdrant_client import ensure_payload_indexes

    log.info("Starting up: loading query encoder...")
    encoder = QueryEncoder.instance()
    encoder.warmup()
    log.info("Query encoder ready.")

    try:
        ensure_payload_indexes()
    except Exception as exc:
        log.warning("Qdrant payload index bootstrap failed: %s", exc)

    log.info("DB engine ready.")
    yield
    log.info("Shutting down: disposing DB engine.")
    if engine is not None:
        engine.dispose()


app = FastAPI(lifespan=lifespan)
install_request_logger(app)
app.include_router(api)
