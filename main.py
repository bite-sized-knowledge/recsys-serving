from fastapi import FastAPI

app = FastAPI()

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# 기본 엔드포인트
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}