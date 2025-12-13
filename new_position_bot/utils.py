import os
import logging
from typing import Optional

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    val = os.getenv(name, default)
    if required and not val:
        raise ValueError(f"Environment variable {name} is required but not set.")
    return val

def get_private_key() -> str:
    """
    Retrieves the Kalshi RSA private key.
    Priority:
    1. Local file 'kalshi_private_key' (for local dev)
    2. GCP Secret Manager (if configured and file not found)
    """
    # 1. Try local file
    local_key_path = "kalshi_private_key"
    if os.path.exists(local_key_path):
        logger.info("Loading private key from local file.")
        with open(local_key_path, "r") as f:
            return f.read()

    # 2. Try GCP Secret Manager
    # Assumes GCP_PROJECT and SECRET_ID are known or fixed convention, 
    # but usually we pass the full resource ID or just the key content in env for simple setups.
    # However, prompt specifically asked to read from Secret Manager on GCP.
    
    # We use a specific Env Var to signal the Secret Resource ID if needed, 
    # or assume a standard name if running in GCP.
    project_id = os.getenv("GCP_PROJECT_ID")
    secret_name = os.getenv("KALSHI_PRIVATE_KEY_SECRET_NAME", "kalshi_private_key")
    
    if project_id:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            logger.info(f"Attempting to load private key from Secret Manager: {name}")
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.warning(f"Failed to fetch from Secret Manager: {e}")
            
    # Fallback: check if the actual key is in an ENV var (common for some deployments)
    key_env = os.getenv("KALSHI_PRIVATE_KEY_CONTENT")
    if key_env:
        return key_env.replace('\\n', '\n') # Handle escaped newlines

    raise RuntimeError("Could not find Kalshi Private Key in local file or Secret Manager.")
