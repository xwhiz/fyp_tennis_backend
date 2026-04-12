from fastapi import APIRouter, Depends

from src.config import settings
from src.core.court_reference import CourtReference
from src.dependencies.auth import AuthContext, get_auth_context

router = APIRouter(tags=["misc"])


@router.get("/")
def test_hello_world():
    return {"success": True, "message": "Hello world"}

@router.get("/api-version")
def get_api_version():
    return {"success": True, "message": "API Version", "version": settings.api_version}

@router.get("/check-health")
def check_health():
    return {"success": True, "message": "OK"}


@router.get("/court_reference")
def get_court_reference(auth_ctx: AuthContext = Depends(get_auth_context)):
    court_reference = CourtReference()
    return {"success": True, "data": court_reference.to_dict()}
