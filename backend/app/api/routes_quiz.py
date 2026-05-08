from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.request_models import QuizRequest
from app.models.response_models import QuizResponse, MCQQuestion
from app.llm.quiz_generator import generate_quiz, QuizGenerationError
from app.core.embeddings import get_embedding_engine
from app.core.vector_store import ChromaVectorStore

router = APIRouter(prefix="/quiz", tags=["quiz"])

@router.post("", response_model=QuizResponse)
async def create_quiz(request: QuizRequest):
    # Validate doc_id exists in ChromaDB
    try:
        engine = get_embedding_engine()
        store = ChromaVectorStore(engine, collection_name=request.doc_id)
        
        if store.count == 0:
            raise HTTPException(
                status_code=404, 
                detail="Document not found. Please upload it first."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        raise HTTPException(
            status_code=404, 
            detail="Document not found. Please upload it first."
        )

    # Call generate_quiz
    try:
        raw_questions = await generate_quiz(
            doc_id=request.doc_id,
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
    except QuizGenerationError as e:
        logger.error(f"QuizGenerationError: {e}")
        # Return 400 Bad Request since this is a user error (e.g., topic not in document)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during quiz generation: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during quiz generation.")

    # Map to MCQQuestion objects
    questions = []
    for i, q in enumerate(raw_questions, start=1):
        questions.append(
            MCQQuestion(
                id=i,
                question=q.get("question", ""),
                options=q.get("options", {}),
                correct=q.get("correct", ""),
                explanation=q.get("explanation", ""),
                source_chunks=q.get("source_chunks", [])
            )
        )

    return QuizResponse(
        doc_id=request.doc_id,
        topic=request.topic,
        questions=questions
    )
