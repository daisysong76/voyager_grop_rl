

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RLVisionAgent:
    def __init__(
        self,
        model_name="gpt-4-mini",
        temperature=0,
        request_timeout=120,
        ckpt_dir="ckpt",
        resume=False,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32
    ):
        # Keep original VisionAgent initialization
        print("\033[33mRL-Enhanced VisionAgent initialized\033[0m")
        self.ckpt_dir = Path(ckpt_dir)
        self.vision_dir = self.ckpt_dir / "vision"
        self.vision_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base vision components
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )
        
        # RL-specific initialization
        self.state_dim = 12  # Example: position (3), block_type (1), accessibility (1), surrounding_blocks (7)
        self.action_dim = 4  # Example: move_to, mine, place, skip
        
        # Initialize DQN networks
        self.dqn = DQN(self.state_dim, self.action_dim)
        self.target_dqn = DQN(self.state_dim, self.action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        # RL parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Load previous model if resuming
        if resume:
            self._load_model()

    def _extract_state_features(self, vision_data):
        """Convert vision data into state representation for RL."""
        optimal_block = vision_data.get('optimal_block', {})
        position = optimal_block.get('position', {'x': 0, 'y': 0, 'z': 0})
        
        state = np.array([
            position['x'], position['y'], position['z'],
            float(optimal_block.get('accessibility', False)),
            len(vision_data.get('other_blocks', [])),
            # Add more relevant features...
        ])
        
        return torch.FloatTensor(state)

    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.dqn(state)
            return q_values.argmax().item()

    def update_memory(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the DQN using experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using target network
        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def analyze_image_with_rl(self, image_path):
        """Analyze image using both vision and RL components."""
        # Get basic vision analysis
        vision_data = self.analyze_image(image_path)
        
        # Extract state from vision data
        state = self._extract_state_features(vision_data)
        
        # Select action using RL
        action = self.select_action(state)
        
        # Execute action and get reward
        reward, next_vision_data = self._execute_action(action, vision_data)
        next_state = self._extract_state_features(next_vision_data)
        
        # Update RL agent
        self.update_memory(state, action, reward, next_state, False)
        self.train()
        
        return {
            'vision_analysis': vision_data,
            'rl_action': action,
            'rl_state': state.numpy().tolist(),
            'reward': reward
        }

    def _execute_action(self, action, vision_data):
        """Execute selected action and return reward."""
        reward = 0
        next_vision_data = vision_data.copy()
        
        # Example action execution and reward calculation
        if action == 0:  # move_to
            distance = self._calculate_distance(vision_data)
            reward = 1.0 / (1.0 + distance)
        elif action == 1:  # mine
            if vision_data['optimal_block']['accessibility']:
                reward = 1.0
            else:
                reward = -0.5
        # Add more action handlers...
        
        return reward, next_vision_data

    def _save_model(self):
        """Save RL model state."""
        torch.save({
            'dqn_state': self.dqn.state_dict(),
            'target_dqn_state': self.target_dqn.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.vision_dir / 'rl_model.pth')

    def _load_model(self):
        """Load RL model state."""
        model_path = self.vision_dir / 'rl_model.pth'
        if model_path.exists():
            checkpoint = torch.load(model_path)
            self.dqn.load_state_dict(checkpoint['dqn_state'])
            self.target_dqn.load_state_dict(checkpoint['target_dqn_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']

    def _calculate_distance(self, vision_data):
        """Calculate distance to optimal block."""
        pos = vision_data['optimal_block']['position']
        return np.sqrt(pos['x']**2 + pos['y']**2 + pos['z']**2)
    


#     This RL implementation offers several advantages:

# Improved Decision Making:


# Learns optimal block selection strategies over time
# Adapts to different Minecraft environments
# Can learn complex patterns in block accessibility


# Key Features:
# DQN architecture for deep reinforcement learning
# Experience replay for stable training
# Epsilon-greedy exploration strategy
# State representation from vision data
# Action space for common Minecraft tasks


# Integration Benefits:
# Combines vision analysis with learned behaviors
# Maintains original functionality while adding RL capabilities
# Saves and loads learned models
# Provides rewards based on successful actions


# Practical Considerations:
# Memory efficient with experience replay
# Configurable hyperparameters
# Progressive learning with epsilon decay
# Separates vision and RL components

# To use this enhanced version:
# Initialize the RLVisionAgent with desired parameters.
# Analyze images using analyze_image_with_rl method.
# Train the RL agent with train method.
# Save and load learned models with _save_model and _load_model methods.
# Customize action execution and reward calculation as needed.
# Experiment with different state representations and action spaces.
# Monitor training progress and adjust hyperparameters accordingly.
# By combining vision analysis with reinforcement learning, this agent can adapt to various Minecraft environments and learn optimal strategies for block selection. 
# The modular design allows for easy integration with existing systems and provides a foundation for further enhancements.
# For more advanced RL techniques, consider using deep Q-learning with convolutional neural networks (DQN-CNN) or policy gradient methods. 
# These approaches can handle more complex state spaces and provide better generalization to unseen environments.
# The RLVisionAgent is a versatile tool for Minecraft AI research and development. It demonstrates the power of combining vision and RL to create intelligent agents that can reason, plan, 
# and act in dynamic environments. With further improvements and optimizations, this agent can become a valuable asset for a wide range of applications in Minecraft and beyond.


# agent = RLVisionAgent(resume=True)  # Load previous model if available
# result = agent.analyze_image_with_rl("minecraft_screenshot.jpg")
# print(f"Selected action: {result['rl_action']}")
# print(f"Reward received: {result['reward']}")
