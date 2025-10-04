from pydantic import BaseModel


class ProcessVideoResponse(BaseModel):
    success: bool
    message: str
    data: dict
