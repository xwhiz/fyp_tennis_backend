from pydantic import BaseModel, Field


class FriendRequestCreate(BaseModel):
    atTag: str = Field(..., min_length=1, description="Opponent @ tag, with or without leading @")
