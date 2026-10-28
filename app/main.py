from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import HRBot

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Chatbot RH (RAG)", version="1.0.0")
bot = HRBot(ROOT / "data" / "policies")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    file: str
    section: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return bot.answer(req.question.strip())


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
