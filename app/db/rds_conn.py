import ssl

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import config

engine = None
SessionLocal = None

def get_engine():
    global engine
    if engine is None:
        DATABASE_URL = (
            f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}"
            f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        )
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            connect_args={"ssl": ssl_ctx},
        )
    
    return engine

def get_sessionmaker():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return SessionLocal