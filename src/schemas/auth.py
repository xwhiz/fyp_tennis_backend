from pydantic import BaseModel, Field


class SignUpRequest(BaseModel):
    firstName: str = Field(min_length=1)
    lastName: str = Field(min_length=1)
    playerHeight: float | None = None
    dominantHand: str = Field(min_length=1)
    email: str = Field(min_length=3)
    password: str = Field(min_length=6)
    consent: bool


class SignInRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=3)


class RefreshTokenRequest(BaseModel):
    token: str = Field(min_length=1)


class ResetPasswordRequest(BaseModel):
    currentPassword: str = Field(min_length=1)
    newPassword: str = Field(min_length=6)
