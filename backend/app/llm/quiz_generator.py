from typing import List, Dict, Any
from loguru import logger

from app.core.embeddings import get_embedding_engine
from app.core.vector_store import ChromaVectorStore
from app.core.retriever import AdvancedRetriever
from app.llm import ollama_client
from app.llm import prompts
from app.llm.output_parser import extract_json, ParseError
from app.llm.validators import validate_mcq_list, ValidationError

class QuizGenerationError(Exception):
    """Exception raised when quiz generation fails after all retries or if no context is found."""
    pass

async def generate_quiz(doc_id: str, topic: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
    # Step 1: Initialize and call retriever to get context chunks
    embedding_engine = get_embedding_engine()
    vector_store = ChromaVectorStore(embedding_engine, collection_name=doc_id)
    retriever = AdvancedRetriever(vector_store, embedding_engine)
    
    # --- Force log the top 3 raw scores before any filtering ---
    try:
        raw_query_res = vector_store.query(topic, n_results=3)
        parsed_raw = AdvancedRetriever._parse_chroma_results(raw_query_res)
        logger.info(f"--- RAW TOP 3 SCORES FOR '{topic}' ---")
        for idx, r in enumerate(parsed_raw, start=1):
            logger.info(f"Raw Candidate {idx}: {r.score:.4f}")
        logger.info("---------------------------------------")
    except Exception as e:
        logger.error(f"Failed to log raw scores: {e}")

    # Using 'threshold' mode ensures that if the user inputs gibberish, 
    # the similarity score will be very low and it will return 0 chunks.
    chunks = retriever.retrieve(topic, mode='threshold', top_k=8, threshold=0.35)
    
    logger.info(f"Retrieved {len(chunks)} chunks for topic '{topic}' (doc_id: {doc_id})")
    
    # Log the similarity score of each matched chunk to the terminal
    for chunk in chunks:
        logger.info(f"Chunk Rank {chunk.rank} - Similarity Score: {chunk.score:.4f}")
        
    if not chunks:
        raise QuizGenerationError(f"No relevant content found for topic: {topic}")
        
    # Step 2: Build prompt
    user_prompt = prompts.build_user_prompt(
        topic=topic,
        chunks=chunks,
        num_questions=num_questions,
        difficulty=difficulty
    )
    
    # Step 3: Generation and validation retry loop
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Starting quiz generation attempt {attempt}/{max_attempts} for topic '{topic}'")
        try:
            # Step 4: Call ollama_client.generate()
            # Defaulting to llama3.1:8b as the model based on system environment context
            raw_response = await ollama_client.generate(
                prompt=user_prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                model="llama3.1:8b"
            )
            
            # Step 5: Extract JSON
            parsed_result = extract_json(raw_response)
            
            # Step 6: Validate MCQ List
            validated_list = validate_mcq_list(parsed_result)
            
            if not validated_list:
                raise QuizGenerationError(f"No valid questions could be generated. This usually means the topic '{topic}' is unrelated to the document.")
            
            logger.info(f"Successfully generated and validated quiz on attempt {attempt}.")
            return validated_list
            
        except (ParseError, ValidationError) as e:
            logger.warning(f"Generation attempt {attempt} failed due to {e.__class__.__name__}: {str(e)}")
            if attempt == max_attempts:
                logger.error(f"All {max_attempts} attempts failed for topic '{topic}'.")
                raise QuizGenerationError(f"Failed to generate quiz after {max_attempts} attempts: {str(e)}") from e
