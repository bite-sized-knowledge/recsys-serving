from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.conn import engine
from app.api import api

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: DB engine is ready.")
    yield
    print("Shutting down: DB engine will be disposed.")
    engine.dispose()

app = FastAPI(lifespan=lifespan)
app.include_router(api)