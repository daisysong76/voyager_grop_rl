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
    "client_id": "3ba45205-c9c0-46bc-8216-4b481f55873d",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}

# voyager = Voyager(
#     #mc_port="55612",
#     azure_login=azure_login,
#     openai_api_key=openai_api_key,
#     ckpt_dir="/Users/daisysong/Desktop/Voyager2/checkpoints", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
#     #resume = True,
#     pause_on_think=False,
# )

# # start lifelong learning
# try:
#     voyager.learn(reset_env=False)
# except KeyboardInterrupt:
#     print("Program interrupted. Stopping mineflayer.")
#     # Add any cleanup code here if necessary

# Initialize the first bot
bot_0 = Voyager(
    bot_id=0,  # Unique ID for the first bot
    mc_port=54961,
    #azure_login=azure_login,
    openai_api_key=openai_api_key,
    max_iterations=100,
    #ckpt_dir="./ckpt",
)

# Initialize the second bot
bot_1 = Voyager(
    bot_id=1,  # Unique ID for the second bot
    mc_port=54961,  # Ensure it connects to the same Minecraft server
    #azure_login=azure_login,
    openai_api_key=openai_api_key,
    max_iterations=100,
    #ckpt_dir="./ckpt",
)

# Define what each bot thread should do
def run_bot(voyager):
    try:
        voyager.learn(reset_env=False)  # Run the bot's learning process
    except Exception as e:
        print(f"Error: {str(e)}")
# Create thread objects - they don't start running yet
bot1_thread = threading.Thread(target=run_bot, args=(bot_0,))
bot2_thread = threading.Thread(target=run_bot, args=(bot_1,))

# Start the threads - this actually starts the bots running
bot1_thread.start()
time.sleep(5)  # Wait before starting second bot
bot2_thread.start()

# Wait for both threads to complete
bot1_thread.join()
bot2_thread.join()


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