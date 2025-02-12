# Multi-Stage Training Process
# DeepSeek's training process involves multiple stages:
# Initial large-scale reinforcement learning
# Supervised fine-tuning on synthetic data
# Further reinforcement learning on reasoning tasks
# Rejection sampling for optimization
# Final reinforcement learning for alignment

# The VisionAgent class is responsible for managing the training and decision-making process using reinforcement learning.
# The multi_stage_training method orchestrates the different stages of training, including initial large-scale reinforcement learning, supervised fine-tuning, further reinforcement learning, rejection sampling optimization, and final reinforcement learning.
# The _supervised_fine_tuning and _rejection_sampling_optimization methods are placeholders for implementing the specific logic for those stages.
# The analyze_image method uses the trained reinforcement learning model to make decisions based on the input image.
# The _preprocess_image method is a placeholder for implementing image preprocessing logic.
# The _perform_analysis method is a placeholder for implementing the analysis logic based on the chosen action.
# pip install stable-baselines3\[extra\]

# python -m voyager.agents.vision
import logging
import openai
import anthropic
import sys

import torch
sys.path.append('/Users/daisysong/Desktop/CS194agent/Voyager_OAI/voyager/')
#import voyager.utils as U
import utils as U
#from voyager import utils as U
import requests
import numpy as np
import os
import json
import base64
from datetime import datetime
# Import ChatOpenAI and HumanMessage from LangChain or the relevant library
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
#import pdb; pdb.set_trace()
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# TODO 4: Import the VisionEnv class from the vision module
# class GRPOPolicy(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(GRPOPolicy, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim)
#         )
    
#     def forward(self, x):
#         return self.fc(x)
class GRPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GRPOPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
class GRPOAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        self.policy = GRPOPolicy(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, rewards, log_probs):
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=1000):
        for i_episode in range(num_episodes):
            state = env.reset()
            rewards = []
            log_probs = []
            
            for t in range(1000):  # Assume max 1000 steps per episode
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                if done:
                    break
                
                state = next_state
            
            self.update_policy(rewards, log_probs)
            
            if i_episode % 10 == 0:
                print(f'Episode {i_episode}\tAverage reward: {sum(rewards)/len(rewards)}')

class EnhancedActionAgent(ActionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpo_agent = GRPOAgent(input_dim=224*224*3, output_dim=4)  # Adjust dimensions as needed

    def process_action(self, events):
        # Use GRPO agent for decision making
        state = self._preprocess_state(events)
        action, _ = self.grpo_agent.select_action(state)
        
        # Combine GRPO decision with existing logic
        grpo_decision = self._interpret_grpo_action(action)
        
        # Existing ActionAgent logic
        existing_decision = super().process_action(events)
        
        # Combine decisions (implement your own logic here)
        final_decision = self._combine_decisions(grpo_decision, existing_decision)
        
        return final_decision

    def _preprocess_state(self, events):
        # Convert events to a state representation for GRPO
        # Implement your own preprocessing logic
        pass

    def _interpret_grpo_action(self, action):
        # Convert GRPO action to a format compatible with ActionAgent
        # Implement your own interpretation logic
        pass

    def _combine_decisions(self, grpo_decision, existing_decision):
        # Implement logic to combine GRPO and existing decisions
        pass

    def train_grpo(self, env, num_episodes=1000):
        self.grpo_agent.train(env, num_episodes)
# end of TODO 4
# prompt engineering: https://www.perplexity.ai/search/we-will-hold-a-series-of-offic-wRm6SoDtQVGxu0wIQCE6WA
# To use this implementation:
# Initialize the EnhancedActionAgent:
# python
# agent = EnhancedActionAgent()
# Train the GRPO component:
# python
# env = YourMinecraftEnvironment()  # Implement this
# agent.train_grpo(env, num_episodes=1000)
# Use the enhanced agent for decision-making:
# python
# action = agent.process_action(events)
# This approach allows you to leverage both the existing ActionAgent's capabilities and the GRPO-based decision-making, potentially improving the agent's performance in the Minecraft environment.

class VisionAgent:
    def __init__(
        self,
        #model_name="gpt-4-turbo", 
        model_name="gpt-4-mini", 
        # TODO 2: how to debug if this following vision_agent was called?
        temperature=0, 
        request_timeout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        print("\033[33mVisionAgent initialized\033[0m")  # Yellow color
        self.ckpt_dir = Path(ckpt_dir)
        self.chat_log = chat_log
        self.execution_error = execution_error

        U.f_mkdir(f"{ckpt_dir}/vision")
        self.vision_dir = self.ckpt_dir / "vision"
        #Define the vision directory using Path
        self.vision_dir = self.ckpt_dir / "vision"
        self.vision_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[34mEnsured directory exists: {self.vision_dir}\033[0m")  # Blue color for directory creation

         #TODO 4: Initialize RL components
        self.env = DummyVecEnv([lambda: VisionEnv()])
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        self.policy = GRPOPolicy(input_dim=224*224*3, output_dim=4)  # Assuming 224x224 RGB images and 4 actions
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.gamma = 0.99
        # end of TODO 4
        
        if resume:
            memory_file = self.vision_dir / "vision_memory.json"
            if memory_file.exists():
                print(f"\033[32mLoading Vision Agent from {memory_file}\033[0m")
                self.vision_memory = self._load_json(memory_file)
                print(f"\033[32mLoaded vision_memory.json successfully in vision.py.\033[0m")
            else:
                self.vision_memory = {}
                print(f"\033[31mvision_memory.json not found. Initializing empty vision_memory.\033[0m")  # Red color
        else:
            self.vision_memory = {}
            print("\033[33mInitialized empty vision_memory.\033[0m")  # Yellow color

        # if resume:
        #     print(f"\033[32mLoading Vision Agent from {ckpt_dir}/vision\033[0m")
        #     self.vision_memory = U.load_json(f"{ckpt_dir}/vision/vision_memory.json")
        #     print(f"\033[32mLoaded vision_memory.json successfully in vision.py.\033[0m")
        # else:
        #     self.vision_memory = {}
        #     print("\033[33mInitialized empty vision_memory.\033[0m")  # Yellow color

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )

    def _load_json(self, file_path):
        """Load JSON from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json(self, data, file_path):
        """Save JSON to a file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
   
    def update_vision_memory(self, vision_data):
        """
        Update vision memory with new data and save to file.

        Args:
            vision_data (dict): New vision data to add.
        """
        logging.info("Updating vision memory.")
        print(f"\033[33mVisionAgent.update_vision_memory called with: {vision_data}\033[0m")

        # Add a timestamp to the new data
        timestamp = datetime.utcnow().isoformat()
        vision_data["timestamp"] = timestamp

        # Path to vision memory file
        #memory_file_path = self.ckpt_dir / "vision" / "vision_memory.json"
        memory_file_path = self.vision_dir / "vision_memory.json"

        # Load existing memory if the file exists
        if memory_file_path.exists():
            try:
                self.vision_memory = self._load_json(memory_file_path)
                if not isinstance(self.vision_memory, dict):
                    logging.error(f"Expected vision_memory.json to be a dict, but got {type(self.vision_memory)}")
                    raise TypeError("vision_memory.json should contain a dictionary.")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {memory_file_path}: {e}")
                raise
        else:
            self.vision_memory = {}

        # Add the new data with the timestamp as the key
        self.vision_memory[timestamp] = vision_data

        # Save the updated memory back to the JSON file
        self._save_json(self.vision_memory, memory_file_path)

        logging.info("New analysis added to vision_memory.json.")
        print("\033[32mNew analysis added to vision_memory.json\033[0m")  # Green color


    # def analyze_image(self, image_path):
    #     """Send the image to GPT-4 Vision or similar models for analysis."""
    #     print(f"\033[33mVisionAgent.analyze_image called with: {image_path}\033[0m")

    #     # Function to encode the image
    #     def encode_image(image_path):
    #         with open(image_path, "rb") as image_file:
    #             return base64.b64encode(image_file.read()).decode('utf-8')

    #     # Encode the image to a base64 string
    #     base64_image = encode_image(image_path)

    #     # Define the response format
    #     response_format = """
    #     {
    #         "optimal_block": {
    #             "type": "string",
    #             "position": {
    #                 "x": "float",
    #                 "y": "float",
    #                 "z": "float"
    #             },
    #             "accessibility": "boolean"
    #         },
    #         "other_blocks": [
    #             {
    #                 "type": "string",
    #                 "position": {
    #                     "x": "float",
    #                     "y": "float",
    #                     "z": "float"
    #                 },
    #                 "accessibility": "boolean"
    #             }
    #         ],
    #         "spatial_reasoning": {
    #             "relative_positions": [
    #                 {
    #                     "block": "string",
    #                     "relative_position": {
    #                         "x": "float",
    #                         "y": "float",
    #                         "z": "float"
    #                     }
    #                 }
    #             ]
    #         }
    #     }
    #     """

    #     # Define the prompt
    #     prompt = f"""
    #     You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
    #     Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

    #     ### Task Instructions
    #     1. **Enhanced Block Detection**:
    #     - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
    #     - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
    #     - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
    #     - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

    #     2. **JSON Requirements**:
    #     - Return your output **strictly as a valid JSON object**.
    #     - Do not include any explanations, comments, or text outside the JSON structure.
    #     - The JSON must follow the format exactly as specified below.

    #     ### RESPONSE FORMAT:
    #     {response_format}

    #     Only respond with the JSON object. Do not include anything else in your output.
    #     """
        
    #    # Prepare messages using LangChain's BaseMessage classes
    #     messages = [
    #         HumanMessage(content=prompt),
    #         HumanMessage(content=f"Analyze the following image (base64 encoded): data:image/jpeg;base64,{base64_image}")
    #     ]

    #     #try:
    #         # Invoke the API
    #         # response = json.loads(self.llm.invoke(messages).content)
    #         # print(response)

    #         # # Handle the response
    #         # if isinstance(response, HumanMessage):
    #         #     response_data = response.content
    #         # elif isinstance(response, str):
    #         #     response_data = json.loads(response)  # Parse JSON if string
    #         # else:
    #         #     response_data = response  # Use as-is if already a dictionary

    #         # # Save the response to a JSON file
    #         # output_dir = os.path.join(self.ckpt_dir, "vision")
    #         # os.makedirs(output_dir, exist_ok=True)
    #         # print(f"\033[33mVisionAgent.analyze_image saving response to: {output_file_path}\033[0m")
    #         # output_file_path = os.path.join(output_dir, "vision_memory.json")

    #         # with open(output_file_path, 'w') as json_file:
    #         #     json.dump(response_data, json_file, indent=4)

    #         # return response_data
    #     try:
    #         # Invoke the model
    #         response = self.llm.invoke(messages).content
    #         response_data = json.loads(response)
    #         print(f"\033[33mVisionAgent.analyze_image response: {response_data}\033[0m")
    #         print(f"\033[33mVisionAgent.analyze_image response type: {type(response_data)}\033[0m")
            
    #         # Ensure that the expected fields are present and of the correct type
    #         if "optimal_block" in response_data:
    #             optimal_block = response_data["optimal_block"]
    #             if isinstance(optimal_block["position"], dict):
    #                 # Ensure position values are numeric
    #                 for key in ["x", "y", "z"]:
    #                     if not isinstance(optimal_block["position"].get(key), (int, float)):
    #                         raise ValueError(f"Expected numeric value for position '{key}', got {optimal_block['position'].get(key)}")

    #         response_data["image_path"] = image_path
    #         self.update_vision_memory(response_data)
    #         print(f"\033[33mVisionAgent.analyze_image updated vision_memory.json\033[0m")

    #         return response_data
    #     except Exception as e:
    #         raise RuntimeError(f"API call failed: {e}")     
    def analyze_image(self, image_path):
        """Send the image to GPT-4 Vision or similar models for analysis."""
        print(f"\033[33mVisionAgent.analyze_image called with: {image_path}\033[0m")

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Encode the image to a base64 string
        base64_image = encode_image(image_path)

        # Define the response format
        response_format = """
        {
            "optimal_block": {
                "type": "string",
                "position": {
                    "x": "float",
                    "y": "float",
                    "z": "float"
                },
                "accessibility": "boolean"
            },
            "other_blocks": [
                {
                    "type": "string",
                    "position": {
                        "x": "float",
                        "y": "float",
                        "z": "float"
                    },
                    "accessibility": "boolean"
                }
            ],
            "spatial_reasoning": {
                "relative_positions": [
                    {
                        "block": "string",
                        "relative_position": {
                            "x": "float",
                            "y": "float",
                            "z": "float"
                        }
                    }
                ]
            }
        }
        """

        # Define the prompt
        prompt = f"""
        You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
        Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

        ### Task Instructions
        1. **Enhanced Block Detection**:
        - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
        - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
        - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
        - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

        2. **JSON Requirements**:
        - Return your output **strictly as a valid JSON object**.
        - Do not include any explanations, comments, or text outside the JSON structure.
        - The JSON must follow the format exactly as specified below.

        ### RESPONSE FORMAT:
        {response_format}

        Only respond with the JSON object. Do not include anything else in your output.
        """
        
       # Prepare messages using LangChain's BaseMessage classes
        messages = [
            HumanMessage(content=prompt),
            HumanMessage(content=f"Analyze the following image (base64 encoded): data:image/jpeg;base64,{base64_image}")
        ]

        try:
            # Invoke the model
            response = self.llm.invoke(messages).content
            response_data = json.loads(response)
            print(f"\033[33mVisionAgent.analyze_image response: {response_data}\033[0m")
            print(f"\033[33mVisionAgent.analyze_image response type: {type(response_data)}\033[0m")

             # Ensure that the expected fields are present and of the correct type
            if "optimal_block" in response_data:
                optimal_block = response_data["optimal_block"]
                if isinstance(optimal_block["position"], dict):
                    # Ensure position values are numeric
                    for key in ["x", "y", "z"]:
                        if isinstance(optimal_block["position"].get(key), str):
                            # Convert to float if the value is a string
                            optimal_block["position"][key] = float(optimal_block["position"][key])

            response_data["image_path"] = image_path
            self.update_vision_memory(response_data)
            print(f"\033[33mVisionAgent.analyze_image updated vision_memory.json\033[0m")

            return response_data
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise RuntimeError(f"API call failed: {e}")     

    def render_vision_message(self, image_path):
        #return HumanMessage(content=self.prompt)
        pass
    def process_vision_message(self, message):
        pass
    def get_insights(self, voxels, block_records, entities):
        pass
    def get_vision_memory(self):
        return self.vision_memory
    def get_vision_memory_by_timestamp(self, timestamp):
        return self.vision_memory.get(timestamp, {})
    
    def get_vision_memory_by_block_type(self, block_type):
        return [data for data in self.vision_memory.values() if data.get("optimal_block", {}).get("type") == block_type]
   
    #TODO 4: Implement the multi_stage_training method
    def multi_stage_training(self, total_timesteps=100000):
        print("Stage 1: Initial large-scale reinforcement learning")
        self.rl_model.learn(total_timesteps=total_timesteps)

        print("Stage 2: Supervised fine-tuning on synthetic data")
        self._supervised_fine_tuning()

        print("Stage 3: Further reinforcement learning on reasoning tasks")
        self.rl_model.learn(total_timesteps=total_timesteps // 2)

        print("Stage 4: Rejection sampling for optimization")
        self._rejection_sampling_optimization()

        print("Stage 5: Final reinforcement learning for alignment")
        self.rl_model.learn(total_timesteps=total_timesteps // 4)

    # Implement supervised fine-tuning on synthetic data
    def _supervised_fine_tuning(self):
        # Generate or load synthetic data
        train_data, val_data = self._get_synthetic_data()
        
        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 3
        for epoch in range(10):  # Adjust number of epochs as needed
            self.model.train()
            for batch in train_data:
                optimizer.zero_grad()
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            val_loss = self._validate(val_data)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 3
            else:
                patience -= 1
                if patience == 0:
                    break  # Early stopping
        
        print(f"Fine-tuning completed. Best validation loss: {best_val_loss}")

        pass

    # Implement rejection sampling for optimization
    def _rejection_sampling_optimization(self):
        # Generate candidate samples
        num_candidates = 1000
        candidates = self.generate_candidates(num_candidates)
        
        # Evaluate candidates
        scores = self.evaluate_candidates(candidates)
        
        # Perform rejection sampling
        acceptance_threshold = np.percentile(scores, 90)  # Accept top 10%
        accepted_samples = [c for c, s in zip(candidates, scores) if s >= acceptance_threshold]
        
        # Update model with accepted samples
        self.update_model(accepted_samples)

    def generate_candidates(self, num_candidates):
        # Generate diverse candidates using the current model
        return [self.model.generate(temperature=0.7) for _ in range(num_candidates)]

    def evaluate_candidates(self, candidates):
        # Evaluate candidates using a reward model or heuristic
        return [self.reward_model.score(c) for c in candidates]

    def update_model(self, accepted_samples):
        # Fine-tune the model on accepted samples
        self.model.fine_tune(accepted_samples)


    def analyze_image(self, image_path):
        # ... existing code ...

        # Use the trained RL model for decision making
        observation = self._preprocess_image(image_path)
        action, _ = self.rl_model.predict(observation)

        # Use the action to guide the analysis
        response_data = self._perform_analysis(action)

        # ... rest of the existing code ...

    def _preprocess_image(self, image_path):
        # Implement image preprocessing logic
        pass

    def _perform_analysis(self, action):
        # Implement analysis logic based on the chosen action
        pass
