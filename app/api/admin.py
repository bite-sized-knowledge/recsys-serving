"""Internal diagnostics endpoints. Auth required."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import verify_api_key
from app.services import drift

router = APIRouter()


@router.get("/diagnostics/drift", tags=["Admin"])
async def get_drift(_api_key: str = Depends(verify_api_key)):
    """가장 최근 인코더 parity 측정 결과를 반환.

    아직 lifespan 백그라운드 task가 첫 측정을 마치지 않았다면 404. 그 이후엔
    available=true/false + 통계 지표를 노출한다.
    """
    report = drift.latest_report()
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="아직 drift 측정이 수행되지 않았습니다.",
        )
    return report
