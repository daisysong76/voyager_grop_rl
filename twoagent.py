from voyager import Voyager
import threading
import time
import socket
import sys

def check_minecraft_server(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', int(port)))
            return result == 0
    except Exception as e:
        print(f"Error checking Minecraft server: {e}")
        return False

def wait_for_minecraft_server(port, timeout=60):
    print(f"Checking Minecraft server on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_minecraft_server(port):
            print(f"✓ Minecraft server is ready on port {port}")
            return True
        print(f"Waiting for Minecraft server on port {port}...")
        time.sleep(2)
    raise TimeoutError(f"Minecraft server not available on port {port}")

def create_bot(mc_port, server_port, username, api_key):
    try:
        print(f"Creating bot {username} (server port: {server_port})...")
        
        # Add retry logic
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                voyager = Voyager(
                    mc_port=mc_port,
                    openai_api_key=api_key,
                    server_port=server_port,
                    bot_username=username,
                    resume=True
                )
                print(f"✓ Bot {username} created successfully")
                return voyager
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
    except Exception as e:
        print(f"Error creating bot {username}: {e}")
        raise

def run_bot(voyager, username):
    try:
        print(f"Starting bot {username}...")
        voyager.learn()
    except Exception as e:
        print(f"Error in bot {username}: {e}")

def main():
    # Configuration
    mc_port = "59520"  # Your Minecraft server port
    api_key = "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"  # Your OpenAI API key
    
    try:
        # Check if Minecraft server is running
        wait_for_minecraft_server(mc_port)
        
        # Create first bot with retry
        voyager_1 = None
        for attempt in range(3):
            try:
                voyager_1 = create_bot(
                    mc_port=mc_port,
                    server_port=3000,
                    username="bot1",
                    api_key=api_key
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
        if not voyager_1:
            raise Exception("Failed to create first bot after 3 attempts")
        
        # Wait between bot creations
        print("Waiting for first bot to initialize...")
        time.sleep(15)
        
        # Create second bot with retry
        voyager_2 = None
        for attempt in range(3):
            try:
                voyager_2 = create_bot(
                    mc_port=mc_port,
                    server_port=3001,
                    username="bot2",
                    api_key=api_key
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
        if not voyager_2:
            raise Exception("Failed to create second bot after 3 attempts")
        
        # Create threads
        bot1_thread = threading.Thread(
            target=run_bot,
            args=(voyager_1, "bot1"),
            name="Bot1Thread"
        )
        
        bot2_thread = threading.Thread(
            target=run_bot,
            args=(voyager_2, "bot2"),
            name="Bot2Thread"
        )
        
        # Start threads with delay
        print("Starting bot1...")
        bot1_thread.start()
        time.sleep(5)
        
        print("Starting bot2...")
        bot2_thread.start()
        
        # Wait for threads to complete
        bot1_thread.join()
        bot2_thread.join()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
