"""Pydantic request/response models = the API contract documented in docs/LLD.md."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=32, examples=["S0001"])
    question_id: str = Field(..., min_length=1, max_length=64, examples=["FRA_03"])
    skill: str = Field(..., min_length=1, max_length=64, examples=["Fractions"])
    timestamp: Optional[float] = Field(None, description="unix seconds; defaults to now")
    batch: int = Field(-1, description="replay batch number (-1 = live traffic)")


class Features(BaseModel):
    overall_acc: float
    skill_acc: float
    skill_attempts: float
    recent_acc5: float
    hours_since_last: float
    question_difficulty: float


class PredictResponse(BaseModel):
    prediction_id: int
    student_id: str
    question_id: str
    skill: str
    p_correct: float = Field(..., ge=0, le=1)
    model_version: str
    features: Features


class OutcomeRequest(BaseModel):
    prediction_id: Optional[int] = Field(None, description="id returned by /predict")
    student_id: Optional[str] = None
    question_id: Optional[str] = None
    skill: Optional[str] = None
    correct: int = Field(..., ge=0, le=1)
    timestamp: Optional[float] = None


class OutcomeResponse(BaseModel):
    status: str
    attempt_id: int


class TopicMastery(BaseModel):
    skill: str
    mastery: float
    attempts: int
    observed_accuracy: Optional[float]
    weak: bool


class WeakTopicsResponse(BaseModel):
    student_id: str
    threshold: float
    model_version: str
    topics: list[TopicMastery]


class NextQuestionsRequest(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=32)
    n: int = Field(5, ge=1, le=20)


class QuestionSuggestion(BaseModel):
    question_id: str
    skill: str
    p_correct: float
    reason: str


class NextQuestionsResponse(BaseModel):
    student_id: str
    model_version: str
    questions: list[QuestionSuggestion]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    database: bool
