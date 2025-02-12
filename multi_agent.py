from voyager import Voyager
import threading
import time
import signal
import sys
import logging
import os

# https://github.com/MineDojo/Voyager/commit/8eecd3d25d0f025d70130ff3eb5a09f9701edb29

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MultiAgentManager:
    def __init__(self, mc_port, openai_api_key, num_agents=2, base_server_port=3000):
        self.mc_port = mc_port
        self.openai_api_key = openai_api_key
        self.num_agents = num_agents
        self.base_server_port = base_server_port
        self.agents = []
        self.threads = []
        self.logger = logging.getLogger('MultiAgentManager')
        self.shutdown_event = threading.Event()
        
        # Custom bot names
        self.bot_names = ["bot1-alby", "bot2-france"]
        
        # Create necessary directories
        self.create_directories()

    def create_directories(self):
        # Create base directories
        base_dirs = ["logs", "ckpt"]
        for dir_name in base_dirs:
            os.makedirs(dir_name, exist_ok=True)
            
        # Create bot-specific directories
        for bot_name in self.bot_names:
            # Create log directories
            os.makedirs(f"logs/mineflayer_{bot_name}", exist_ok=True)
            os.makedirs(f"logs/minecraft_{bot_name}", exist_ok=True)
            
            # Create checkpoint directories and subdirectories
            os.makedirs(f"ckpt/{bot_name}/action", exist_ok=True)
            os.makedirs(f"ckpt/{bot_name}/curriculum", exist_ok=True)
            os.makedirs(f"ckpt/{bot_name}/skill", exist_ok=True)
            
            # Initialize chest_memory.json if it doesn't exist
            chest_memory_path = f"ckpt/{bot_name}/action/chest_memory.json"
            if not os.path.exists(chest_memory_path):
                with open(chest_memory_path, 'w') as f:
                    f.write('{}')

    def create_agent(self, index):
        server_port = self.base_server_port + index
        try:
            agent = Voyager(
                mc_port=self.mc_port,
                openai_api_key=self.openai_api_key,
                server_port=server_port,
                bot_username=self.bot_names[index],
                resume=True,  # Changed to False for first run
                env_wait_ticks=20,
                env_request_timeout=600,
                reset_placed_if_failed=False,
                max_iterations=160,
                ckpt_dir=f"ckpt/{self.bot_names[index]}"  # Explicitly set checkpoint directory
            )
            self.logger.info(f"Created agent {self.bot_names[index]} with server port {server_port}")
            return agent
        except Exception as e:
            self.logger.error(f"Error in create_agent for {self.bot_names[index]}: {str(e)}")
            raise e

    def run_agent(self, agent):
        try:
            self.logger.info(f"Starting agent {agent.env.bot_username}")
            while not self.shutdown_event.is_set():
                try:
                    agent.learn()
                    if self.shutdown_event.is_set():
                        break
                except Exception as e:
                    self.logger.error(f"Error in agent {agent.env.bot_username}: {str(e)}")
                    if not self.shutdown_event.is_set():
                        time.sleep(5)
        finally:
            try:
                agent.close()
                self.logger.info(f"Closed agent {agent.env.bot_username}")
            except Exception as e:
                self.logger.error(f"Error closing agent {agent.env.bot_username}: {str(e)}")

    def start(self):
        # Create and start agents
        for i in range(self.num_agents):
            try:
                agent = self.create_agent(i)
                self.agents.append(agent)
                time.sleep(5)  # Wait between agent creation
            except Exception as e:
                self.logger.error(f"Error creating agent {i + 1}: {str(e)}")
                continue

        # Create and start threads
        for agent in self.agents:
            thread = threading.Thread(
                target=self.run_agent,
                args=(agent,),
                name=f"Thread-{agent.env.bot_username}"
            )
            self.threads.append(thread)
            thread.start()
            time.sleep(5)  # Wait between thread starts
            self.logger.info(f"Started thread for {agent.env.bot_username}")

    def stop(self):
        self.logger.info("Shutting down all agents...")
        self.shutdown_event.set()
        
        # First close all agents
        for agent in self.agents:
            try:
                agent.close()
            except Exception as e:
                self.logger.error(f"Error closing agent {agent.env.bot_username}: {str(e)}")
        
        # Then terminate threads with a shorter timeout
        for thread in self.threads:
            try:
                thread.join(timeout=5)  # Reduced timeout
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate cleanly")
            except Exception as e:
                self.logger.error(f"Error joining thread {thread.name}: {str(e)}")

        self.logger.info("All agents shut down")

    def signal_handler(self, signum, frame):
        self.logger.info("\nReceived shutdown signal. Stopping agents gracefully...")
        try:
            self.stop()
        finally:
            sys.exit(0)


if __name__ == "__main__":
    # Your existing configuration
    MC_PORT = 61943  # Your Minecraft server port
    NUM_AGENTS = 2  # Number of bots you want to run
    BASE_SERVER_PORT = 3000

    # Create and start the manager
    manager = MultiAgentManager(
        mc_port=MC_PORT,
        openai_api_key=OPENAI_API_KEY,
        num_agents=NUM_AGENTS,
        base_server_port=BASE_SERVER_PORT
    )

    try:
        manager.start()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        manager.stop()
        sys.exit(1)