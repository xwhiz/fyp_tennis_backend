from typing import Any


def success_response(message: str, data: Any | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return response


def error_response(message: str) -> dict[str, Any]:
    return {"success": False, "message": message}
