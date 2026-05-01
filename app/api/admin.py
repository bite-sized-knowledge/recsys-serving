"""Internal diagnostics endpoints. Auth required."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.recommend.service import get_request_metrics
from app.core.auth import verify_api_key
from app.services import drift

router = APIRouter()


@router.get("/diagnostics/recommend", tags=["Admin"])
async def get_recommend_metrics(_api_key: str = Depends(verify_api_key)):
    """프로세스 인메모리 추천 요청 카운터 + latency p50/p95.

    휘발성 (process restart 시 reset). 일별 누적은 recommendation_metric_daily 가 담당.
    backfilled / requests 가 높으면 글로벌 풀 사이즈 / 카테고리 분포 점검 필요.
    """
    return get_request_metrics()


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
