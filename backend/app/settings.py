

from starlette.config import Config
# from starlette.datastructures import Secret

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()

GROQ_API_KEY = config("GROQ_API_KEY", cast=str)
PINECONE_API_KEY=config("PINECONE_API_KEY", cast=str)
OPENAI_API_KEY=config("OPENAI_API_KEY", cast=str)
