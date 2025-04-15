from fastapi import APIRouter, Response, status

router = APIRouter()

@router.get("", tags=["health"], status_code=status.HTTP_200_OK)
async def health_check():
    """
    헬스체크 엔드포인트
    """
    return Response(status_code=status.HTTP_200_OK)