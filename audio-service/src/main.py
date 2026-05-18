from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest, AnalysisResult

app = FastAPI(title="Audio Analysis Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    # Implemented in Task 7
    raise HTTPException(status_code=501, detail="Not yet implemented")
