from pydantic import BaseModel, Field


class UpdateProfileRequest(BaseModel):
    firstName: str = Field(min_length=1)
    lastName: str = Field(min_length=1)
    playerHeight: float | None = None
    dominantHand: str = Field(min_length=1)
