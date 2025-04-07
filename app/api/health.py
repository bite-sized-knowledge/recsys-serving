from fastapi import APIRouter

router = APIRouter()

@router.get("", tags=["health"])
async def health_check():
    """
    헬스체크 엔드포인트
    """
    return {"status": "ok"}