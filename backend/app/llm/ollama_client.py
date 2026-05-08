import json
import httpx
from loguru import logger

try:
    from app.config import OLLAMA_HOST
except ImportError:
    OLLAMA_HOST = "http://localhost:11434"

OLLAMA_TIMEOUT = 300.00

async def generate(prompt: str, system_prompt: str, model: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
            
    except httpx.ConnectError as e:
        logger.error(f"Connection refused: Ollama is not running at {OLLAMA_HOST}. ({e})")
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Timeout: model too slow to respond. ({e})")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: bad response from Ollama. ({e})")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

async def is_available() -> bool:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            return True
    except httpx.ConnectError as e:
        logger.error(f"Connection refused: Ollama is not running at {OLLAMA_HOST}. ({e})")
        return False
    except httpx.TimeoutException as e:
        logger.error(f"Timeout: model too slow to respond. ({e})")
        return False
    except Exception as e:
        logger.error(f"Ollama availability check failed: {e}")
        return False
