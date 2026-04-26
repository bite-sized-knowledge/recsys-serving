from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.rds_conn import engine
from app.api import api
from app.middleware.request_logger import install as install_request_logger
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: DB engine is ready.")
    yield
    print("Shutting down: DB engine will be disposed.")
    engine.dispose()

app = FastAPI(lifespan=lifespan)
install_request_logger(app)
app.include_router(api)