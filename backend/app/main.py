from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.api import routes_health, routes_upload, routes_quiz

app = FastAPI(
    title='Offline RAG Quiz Generator',
    version='1.0.0'
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_health.router)
app.include_router(routes_upload.router)
app.include_router(routes_quiz.router)

@app.on_event("startup")
async def startup_event():
    # Check for CUDA availability
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    logger.info("=== Starting Offline RAG Quiz Generator API ===")
    logger.info(f"Ollama Host: {settings.OLLAMA_HOST}")
    logger.info(f"Ollama Model: {settings.OLLAMA_MODEL}")
    logger.info(f"Chroma Path: {settings.CHROMA_PATH}")
    logger.info(f"CUDA Available: {cuda_available}")
    logger.info("===============================================")

@app.get("/")
async def root():
    return {
        'message': 'Offline RAG Quiz Generator API',
        'docs': '/docs'
    }
