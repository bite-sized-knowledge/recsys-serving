import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from app.core import settings

class Connection:
    def __init__(self):
        # SQLAlchemy 연결 초기화
        self.engine = None
        self.SessionLocal = None

        # DB 연결 시작
        self._connect_to_rds()

    def connect_to_engine(self, host, user, password, database, port):
        """ SQLAlchemy 엔진 생성 """
        DATABASE_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        return create_engine(DATABASE_URL)

    def _connect_to_rds(self):
        try:
            print(f"Connecting to RDS: {settings.DB_HOST}:{settings.DB_PORT}")

            # SQLAlchemy 엔진 생성
            self.engine = self.connect_to_engine(
                host=settings.DB_HOST,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                database=settings.DB_NAME,
                port=settings.DB_PORT
            )

            # 세션 생성기 초기화
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            print("Connected to RDS!")

        except Exception as e:
            print("Error occurred:", e)
            raise

    def execute(self, query):
        """ SELECT 쿼리를 실행하고 Pandas DataFrame으로 반환 """
        if not self.engine:
            raise Exception("No active DB connection.")
        
        # Pandas를 사용해 쿼리 실행 결과를 DataFrame으로 반환
        return pd.read_sql(query, self.engine)

    def _raw_execute(self, query, values=None):
        """ INSERT/UPDATE/DELETE 쿼리를 실행 """
        if not self.engine:
            raise Exception("No active DB connection.")
        
        with self.engine.connect() as connection:
            try:
                connection.execute(text(query), values)
                connection.commit()
                print("Query executed successfully.")
            except Exception as e:
                print(f"Error executing query: {query}, values: {values}. Error: {e}")
                raise

    def close(self):
        """ 연결 종료 """
        if self.engine:
            self.engine.dispose()
            print("SQLAlchemy Engine Disposed.")
