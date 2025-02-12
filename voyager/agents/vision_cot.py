import logging
import openai
import anthropic
import sys
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
class VisionAgent:
    def __init__(self, model_name="gpt-4-mini", temperature=0, request_timeout=120,
                 ckpt_dir="ckpt", resume=False, chat_log=True, execution_error=True):
        print("\033[33mVisionAgent initialized with Chain of Thought reasoning\033[0m")
        self.ckpt_dir = Path(ckpt_dir)
        self.chat_log = chat_log
        self.execution_error = execution_error
        
        # Set up vision directory
        self.vision_dir = self.ckpt_dir / "vision"
        self.vision_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load memory
        if resume:
            memory_file = self.vision_dir / "vision_memory.json"
            if memory_file.exists():
                self.vision_memory = self._load_json(memory_file)
                print("\033[32mLoaded existing vision memory\033[0m")
            else:
                self.vision_memory = {}
                print("\033[31mNo existing memory found, starting fresh\033[0m")
        else:
            self.vision_memory = {}
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )

    def _load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json(self, data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def analyze_image(self, image_path):
        """Analyze image using chain-of-thought reasoning process."""
        print(f"\033[33mStarting chain-of-thought analysis for: {image_path}\033[0m")

        # Step 1: Encode image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        base64_image = encode_image(image_path)

        # Step 2: Define the structured reasoning prompt
        cot_prompt = """
        You are a highly capable assistant analyzing Minecraft visual data. Follow this chain of thought process:

        1) Initial Scene Analysis:
           - What blocks are visible in the scene?
           - What are their approximate positions?
           - Are there any notable patterns or clusters?

        2) Accessibility Assessment:
           - Which blocks can be easily reached?
           - Are there any obstacles or hazards?
           - What is the best approach path?

        3) Optimal Target Selection:
           - Compare distances to each viable block
           - Evaluate accessibility factors
           - Consider clustering for efficiency
           - Determine the single best target

        4) Spatial Context:
           - How do blocks relate to each other?
           - What is the relative positioning?
           - Are there any environmental constraints?

        Based on this analysis, provide a JSON response in this format:
        {
            "reasoning_steps": {
                "scene_analysis": {"observations": [], "conclusions": []},
                "accessibility": {"observations": [], "conclusions": []},
                "target_selection": {"observations": [], "conclusions": []},
                "spatial_context": {"observations": [], "conclusions": []}
            },
            "optimal_block": {
                "type": "string",
                "position": {"x": float, "y": float, "z": float},
                "accessibility": boolean,
                "selection_reason": "string"
            },
            "other_blocks": [{
                "type": "string",
                "position": {"x": float, "y": float, "z": float},
                "accessibility": boolean
            }],
            "spatial_reasoning": {
                "relative_positions": [{
                    "block": "string",
                    "relative_position": {"x": float, "y": float, "z": float}
                }]
            }
        }
        """

        messages = [
            HumanMessage(content=cot_prompt),
            HumanMessage(content=f"Analyze this image: data:image/jpeg;base64,{base64_image}")
        ]

        try:
            # Get response with reasoning steps
            response = self.llm.invoke(messages).content
            response_data = json.loads(response)
            
            # Validate and process the response
            self._validate_response(response_data)
            
            # Add metadata
            response_data["image_path"] = image_path
            response_data["timestamp"] = datetime.utcnow().isoformat()
            
            # Update memory with the reasoned analysis
            self.update_vision_memory(response_data)
            
            print("\033[32mCompleted chain-of-thought analysis\033[0m")
            return response_data

        except Exception as e:
            logging.error(f"Chain-of-thought analysis failed: {e}")
            raise RuntimeError(f"Analysis failed: {e}")

    def _validate_response(self, response_data):
        """Validate the structure and content of the response."""
        required_fields = ["reasoning_steps", "optimal_block", "other_blocks", "spatial_reasoning"]
        
        for field in required_fields:
            if field not in response_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert string coordinates to float if needed
        if "optimal_block" in response_data:
            pos = response_data["optimal_block"]["position"]
            for key in ["x", "y", "z"]:
                if isinstance(pos[key], str):
                    pos[key] = float(pos[key])

    def update_vision_memory(self, vision_data):
        """Update vision memory with new reasoned analysis."""
        print("\033[33mUpdating vision memory with reasoned analysis\033[0m")
        
        memory_file_path = self.vision_dir / "vision_memory.json"
        
        # Load existing memory
        if memory_file_path.exists():
            try:
                self.vision_memory = self._load_json(memory_file_path)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to load existing memory: {e}")
                self.vision_memory = {}
        
        # Add new analysis with timestamp as key
        timestamp = vision_data.get("timestamp", datetime.utcnow().isoformat())
        self.vision_memory[timestamp] = vision_data
        
        # Save updated memory
        self._save_json(self.vision_memory, memory_file_path)
        print("\033[32mSuccessfully saved reasoned analysis to memory\033[0m")

    def get_reasoning_history(self, limit=None):
        """Retrieve the history of reasoning steps from memory."""
        sorted_entries = sorted(self.vision_memory.items(), key=lambda x: x[0], reverse=True)
        if limit:
            sorted_entries = sorted_entries[:limit]
        
        return {timestamp: data.get("reasoning_steps", {}) 
                for timestamp, data in sorted_entries}





# Advantages of using Chain of Thought for this project:

# 1. Complex Spatial Analysis
# - The agent needs to understand multiple blocks, their positions, and relationships
# - Breaking down the reasoning into steps helps ensure nothing is missed
# - Good for handling complex scenes with multiple objects

# 2. Decision Making Transparency
# - The reasoning steps help explain why certain blocks were chosen as "optimal"
# - Makes debugging easier when the agent makes incorrect decisions
# - Helps in understanding why the agent might be failing in certain scenarios

# However, there are significant drawbacks:

# 1. Performance Impact
# - Adding CoT increases token usage significantly
# - Will make each API call more expensive
# - Could introduce latency in what should be real-time gameplay analysis

# 2. Unnecessary Complexity
# - The current task is relatively straightforward: identify blocks and their positions
# - The existing JSON structure already captures the essential information
# - CoT might be overengineering for what's essentially pattern recognition

# 3. Minecraft-Specific Considerations
# - Minecraft gameplay often requires quick, real-time decisions
# - The added processing time from CoT could impact gameplay smoothness
# - The game environment is structured and predictable, reducing the need for complex reasoning

# My Recommendation:
# I would NOT recommend implementing Chain of Thought for this specific project because:

# 1. The current implementation is already well-structured with clear outputs
# 2. The performance cost outweighs the potential benefits
# 3. Real-time gameplay needs favor speed and efficiency over detailed reasoning
# 4. The structured environment of Minecraft doesn't require complex reasoning chains

# Instead, I would suggest:
# - Keep the current direct analysis approach
# - Focus on optimizing performance
# - Add better error logging and debugging information if needed
# - Consider adding simple heuristics for block selection rather than full CoT

# Would you like to explore alternative ways to improve the agent's decision-making without introducing the overhead of CoT?