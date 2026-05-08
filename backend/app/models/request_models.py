from pydantic import BaseModel, Field
from typing import Literal

class QuizRequest(BaseModel):
    doc_id: str
    topic: str
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
