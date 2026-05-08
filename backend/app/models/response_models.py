from pydantic import BaseModel

MCQOption = dict[str, str]

class MCQQuestion(BaseModel):
    id: int
    question: str
    options: dict[str, str]
    correct: str
    explanation: str
    source_chunks: list[str] = []

class QuizResponse(BaseModel):
    doc_id: str
    topic: str
    questions: list[MCQQuestion]

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    status: str

class HealthResponse(BaseModel):
    status: str
    ollama: dict
    chroma: dict
    embeddings: dict
