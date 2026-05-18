from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    storage_key: str = Field(..., min_length=1, max_length=1024)


class AnalysisResult(BaseModel):
    bpm: float
    key: str
    duration_sec: float
    beat_grid: list[float]
    sections: list[dict]
    stems: dict[str, dict]
