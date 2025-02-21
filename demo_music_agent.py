from voyager import Voyager
import os
from dotenv import load_dotenv
import threading  # Import Python's threading module
import time
from voyager_music_agent import VoyagerMusicAgent  # Import the VoyagerMusicAgent

# Load environment variables from .env file (if you're using one)
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
azure_client_id = os.getenv('AZURE_CLIENT_ID')
azure_secret_value = os.getenv('AZURE_SECRET_VALUE')

if not openai_api_key:
    raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")
if not azure_client_id:
    raise ValueError("No client ID found. Please set the AZURE_CLIENT_ID environment variable.")
if not azure_secret_value:
    raise ValueError("No secret value found. Please set the AZURE_SECRET_VALUE environment variable.")

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
# mc_port is only for local testing and will not work in the cloud.
azure_login = {
    "client_id": azure_client_id,
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": azure_secret_value,
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}

voyager = Voyager(
    mc_port="55612",
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    #ckpt_dir="/Users/daisysong/Desktop/Voyager2/checkpoints", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    resume = True,
)

# Initialize the VoyagerMusicAgent
music_agent = VoyagerMusicAgent(voyager)

# Start lifelong learning
try:
    while True:
        # Update music based on game state
        music_agent.update()
        
        # Your regular Voyager agent logic here
        voyager.learn(reset_env=False)
except KeyboardInterrupt:
    print("Program interrupted. Stopping mineflayer.")
    # Add any cleanup code here if necessary