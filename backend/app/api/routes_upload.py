from fastapi import APIRouter, UploadFile, File, HTTPException, status
import uuid
import shutil
from pathlib import Path
from loguru import logger

from app.config import settings
from app.models.response_models import UploadResponse
from app.core.document_loader import PDFDocumentLoader
from app.core.text_cleaner import TextCleaner
from app.core.chunker import SemanticChunker
from app.core.embeddings import get_embedding_engine
from app.core.vector_store import ChromaVectorStore

router = APIRouter(prefix='/upload', tags=['upload'])

@router.post("", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Validate PDF
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Generate doc_id
    doc_id = uuid.uuid4().hex[:9]
    
    # Save the file
    upload_dir = Path(settings.UPLOAD_PATH)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{doc_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"[{doc_id}] Saved uploaded file to {file_path}")
        
        # ── STAGE 1: Document Loading ─────────────────────────────────────────
        logger.info(f"[{doc_id}] Stage 1: Loading PDF document...")
        loader = PDFDocumentLoader(upload_dir)
        raw_documents = loader.load_from_path(file_path)
        
        if not raw_documents:
            raise ValueError("No text could be extracted from the PDF.")
            
        # Add doc_id to metadata so it can be queried later
        for doc in raw_documents:
            doc.metadata["doc_id"] = doc_id
            
        # ── STAGE 2: Text Cleaning ────────────────────────────────────────────
        logger.info(f"[{doc_id}] Stage 2: Cleaning text...")
        cleaner = TextCleaner()
        cleaned_documents = cleaner.clean(raw_documents)
        
        # ── STAGE 3: Embeddings Initialization ────────────────────────────────
        logger.info(f"[{doc_id}] Stage 3: Loading embedding model...")
        embedding_engine = get_embedding_engine()
        _ = embedding_engine.embed_query("warmup")
        
        # ── STAGE 4: Semantic Chunking ────────────────────────────────────────
        logger.info(f"[{doc_id}] Stage 4: Semantic chunking...")
        chunker = SemanticChunker(embedding_engine=embedding_engine)
        chunks = chunker.chunk(cleaned_documents)
        
        # ── STAGE 5: Vector Storage ───────────────────────────────────────────
        logger.info(f"[{doc_id}] Stage 5: Storing vectors in ChromaDB...")
        vector_store = ChromaVectorStore(embedding_engine, collection_name=doc_id)
        chunks_stored = vector_store.add_chunks(chunks)
        
        logger.info(f"[{doc_id}] Pipeline complete. Indexed {len(cleaned_documents)} pages into {chunks_stored} chunks.")
        
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            pages=len(cleaned_documents),
            chunks=chunks_stored,
            status="indexed"
        )
        
    except Exception as e:
        logger.error(f"[{doc_id}] Pipeline failed: {str(e)}")
        # Delete the saved file if it exists
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
