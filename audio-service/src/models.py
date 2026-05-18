from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    storage_key: str


class AnalysisResult(BaseModel):
    bpm: float
    key: str
    duration_sec: float
    beat_grid: list[float]
    sections: list[dict]
    stems: dict[str, dict]
