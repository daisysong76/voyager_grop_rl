import json
import re
import time
import torch
import voyager.utils as U
from javascript import require
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from voyager.agents.vision import VisionAgent
from voyager.prompts import load_prompt
from voyager.control_primitives_context import load_control_primitives_context
from torch.distributions import Categorical


class ActionAgent:
    def __init__(
        self,
        model_name="gpt-4",
        temperature=0,
        request_timeout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
        vision_agent=None,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")

        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )

        if vision_agent is None:
            print("\033[33mActionAgent initializing VisionAgent\033[0m")
            self.vision_agent = VisionAgent()
        else:
            self.vision_agent = vision_agent
        
        print("\033[33mActionAgent getting vision_memory\033[0m")
        self.vision_data = self.vision_agent.get_vision_memory()

    def generate_candidate_actions(self, task, context):
        """
        Generates multiple possible actions for a given task.
        Uses LLM-based Chain of Thought reasoning to suggest diverse action strategies.
        """
        prompt = f"""
        Task: {task}
        Context: {context}
        
        **Step 1: Generate multiple strategies**  
        - List at least 3 action strategies.
        - Consider possible **obstacles** and **alternative paths**.

        **Step 2: Score each action**  
        - Rate strategies based on resource usage, efficiency, and adaptability.

        **Step 3: Justify ranking with Chain of Thought (CoT)**  
        - Explain why one action is superior to others.  
        - Suggest refinements to weaker strategies.

        Return JSON format:
        {{"actions": [{{"strategy": "...", "steps": ["step1", "step2", "..."], "expected_score": float, "reasoning": "...", "constraints": ["...", "..."]}}]}}
        """
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        try:
            return json.loads(response)["actions"]
        except json.JSONDecodeError:
            return []

    def rank_and_select_best_action(self, candidate_actions):
        """
        Uses Group-Relative Policy Optimization (GRPO) to rank action strategies.
        Higher scoring strategies are prioritized, and weaker ones are refined.
        """
        state_tensor = torch.randn(512)  # Placeholder for environment state representation
        action_scores = torch.tensor([action["expected_score"] for action in candidate_actions])

        # Normalize scores for GRPO
        relative_advantage = action_scores - action_scores.mean()
        probabilities = torch.softmax(relative_advantage, dim=-1)

        action_dist = Categorical(probabilities)
        best_action_index = action_dist.sample().item()

        return candidate_actions[best_action_index]

    def self_reflection(self, selected_action):
        """
        Self-reflection with Chain of Thought:
        - Evaluates weaknesses of the selected action.
        - Refines and improves action for better success probability.
        """
        prompt = f"""
        Action Strategy: {selected_action["strategy"]}
        Steps: {selected_action["steps"]}

        **Reflection Questions:**  
        - Are there any risks or inefficiencies?  
        - Could this plan be **optimized for efficiency**?  
        - Would another agent take a **different approach**?  
        - How can this action be improved?  

        **Output JSON Format:**  
        {{"refined_strategy": "...", "refined_steps": ["step1", "step2", "..."], "final_reasoning": "..."}}
        """
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        try:
            refined_action = json.loads(response)
            selected_action["strategy"] = refined_action["refined_strategy"]
            selected_action["steps"] = refined_action["refined_steps"]
            selected_action["reasoning"] = refined_action["final_reasoning"]
        except json.JSONDecodeError:
            pass  # Keep original action if refinement fails

        return selected_action

    def process_action(self, task, context):
        """
        Implements GRPO + Chain of Thought reasoning for action selection:
        1. Generate multiple possible strategies.
        2. Rank & select the best strategy using GRPO.
        3. Perform self-reflection using Chain of Thought.
        4. Return optimized action execution plan.
        """
        candidate_actions = self.generate_candidate_actions(task, context)
        if not candidate_actions:
            return None  # If no candidates, return None
        
        selected_action = self.rank_and_select_best_action(candidate_actions)
        optimized_action = self.self_reflection(selected_action)

        return optimized_action
