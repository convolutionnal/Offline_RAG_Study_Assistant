try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    OLLAMA_HOST: str = 'http://localhost:11434'
    OLLAMA_MODEL: str = 'llama3.1:8b'
    EMBEDDING_MODEL: str = 'all-mpnet-base-v2'
    CHROMA_PATH: str = 'data/chroma_db'
    UPLOAD_PATH: str = 'data/uploads'
    EXTRACTED_PATH: str = 'data/extracted'
    MAX_RETRIES: int = 3
    TOP_K_CHUNKS: int = 8
    MAX_QUESTIONS: int = 20

    class Config:
        env_file = '.env'

settings = Settings()
