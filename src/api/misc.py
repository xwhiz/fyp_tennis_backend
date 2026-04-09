from fastapi import APIRouter, Depends, HTTPException

from src.core.court_reference import CourtReference
from src.dependencies.auth import AuthContext, get_auth_context
from src.models.user import UserRole

router = APIRouter(tags=["misc"])


@router.get("/")
def test_hello_world():
    return {"success": True, "message": "Hello world"}


@router.get("/check-health")
def check_health():
    return {"success": True, "message": "OK"}


@router.get("/court_reference")
def get_court_reference(auth_ctx: AuthContext = Depends(get_auth_context)):
    if auth_ctx.role != UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Access denied")
    court_reference = CourtReference()
    return {"success": True, "data": court_reference.to_dict()}
