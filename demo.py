from voyager import Voyager
import os
from dotenv import load_dotenv
import threading  # Import Python's threading module
import time

# Load environment variables from .env file (if you're using one)
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
# mc_port is only for local testing and will not work in the cloud.
# azure_login = {
#     "client_id": os.getenv("AZURE_CLIENT_ID", "default_client_id"),
#     "redirect_url": "https://127.0.0.1/auth-response",
#     "secret_value":os.getenv("AZURE_SECRET_VALUE", "default_secret_value"),  # Fetch from env,
#     "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
# }
azure_login = {
    "client_id": "your_client_id",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}

voyager = Voyager(
    mc_port="55612",
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    #ckpt_dir="/Users/daisysong/Desktop/Voyager2/checkpoints", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    resume = True,
)

# start lifelong learning
try:
    voyager.learn(reset_env=False)
except KeyboardInterrupt:
    print("Program interrupted. Stopping mineflayer.")
    # Add any cleanup code here if necessary


"""
{
    "OPENAI_API_KEY": "",
    "OPENAI_ORG_ID": "",
    "GEMINI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "REPLICATE_API_KEY": "",
    "GROQCLOUD_API_KEY": "",
    "HUGGINGFACE_API_KEY": "",
    "QWEN_API_KEY":""
}
"""
