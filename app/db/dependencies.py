from sqlalchemy.orm import Session
from app.db.conn import get_sessionmaker

def get_db():
    SessionLocal = get_sessionmaker()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()