from fastapi import APIRouter
from app.models.response_models import HealthResponse
from app.llm import ollama_client
from app.config import settings
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import ChromaVectorStore

router = APIRouter(prefix='/health', tags=['health'])

@router.get("", response_model=HealthResponse)
async def health_check():
    # Check Ollama
    ollama_reachable = await ollama_client.is_available()
    ollama_status = {
        "reachable": ollama_reachable,
        "model": settings.OLLAMA_MODEL
    }
    
    # Check embeddings
    try:
        engine = EmbeddingEngine()
        device = getattr(engine, "_device", "unknown")
        embeddings_status = {
            "loaded": True,
            "device": device
        }
    except Exception as e:
        embeddings_status = {
            "loaded": False,
            "device": "unknown",
            "error": str(e)
        }
        engine = None

    # Check ChromaDB
    try:
        if engine:
            store = ChromaVectorStore(engine)
            count = store.count
            chroma_status = {
                "connected": True,
                "collection_count": count
            }
        else:
            chroma_status = {
                "connected": False,
                "error": "Cannot connect without embedding engine"
            }
    except Exception as e:
        chroma_status = {
            "connected": False,
            "error": str(e)
        }

    # Determine overall status
    if ollama_reachable and chroma_status.get("connected") and embeddings_status.get("loaded"):
        overall_status = "ok"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        ollama=ollama_status,
        chroma=chroma_status,
        embeddings=embeddings_status
    )
