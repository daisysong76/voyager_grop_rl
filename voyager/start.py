import os
from voyager import Voyager

# Use environment variables to load sensitive information
azure_login = {
    "client_id": os.getenv("AZURE_CLIENT_ID", "default_client_id"),  # Fetch from env, fallback to a default if not set
    "redirect_url": "https://127.0.0.1/auth-response",  # Hardcoded URL, since it's probably not sensitive
    "secret_value": os.getenv("AZURE_SECRET_VALUE", "default_secret_value"),  # Fetch from env
    "version": "fabric-loader-0.14.18-1.19",  # Hardcoded version
}

# Fetch OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY", "default_openai_api_key")  # Fetch from env

# Create the Voyager instance with environment values
voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
)

# Start lifelong learning
voyager.learn()
