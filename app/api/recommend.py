from fastapi import APIRouter, Body 

router = APIRouter()

@router.get("", tags=["feeds"])
async def recommend_feeds(member_id: str = Body(..., embed=True)):

    return {
        "articles" : [

        ]
    }