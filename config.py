"""
Central configuration. All secrets are read from environment variables
(never hardcoded) so this project is safe to make public on GitHub.

Locally: create a `.env` file (see .env.example) and this module will
load it automatically via python-dotenv.

On Hugging Face Spaces / Railway / Render: set these as "Secrets" /
environment variables in the platform's dashboard.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file exists (e.g. in production)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cis-department-kb")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "cis-department")

MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL", "https://mcp-server-production-6150.up.railway.app"
)
A2A_SERVER_URL = os.environ.get(
    "A2A_SERVER_URL", "https://a2a-server-production-f1a0.up.railway.app"
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to a .env file locally, "
        "or as a Secret in your deployment platform."
    )
if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not set. Add it to a .env file locally, "
        "or as a Secret in your deployment platform."
    )
