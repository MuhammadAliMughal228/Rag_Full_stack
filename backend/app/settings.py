import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from starlette.config import Config # type: ignore

# Load environment variables from .env file
config = Config(".env")

# Define configuration variables
GROQ_API_KEY = config("GROQ_API_KEY", cast=str)
PINECONE_API_KEY = config("PINECONE_API_KEY", cast=str)
OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str)
PINECONE_ENV = config("PINECONE_ENV", cast=str)

logger.info(f"GROQ_API_KEY: {GROQ_API_KEY}")
logger.info(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
logger.info(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
logger.info(f"PINECONE_ENV: {PINECONE_ENV}")