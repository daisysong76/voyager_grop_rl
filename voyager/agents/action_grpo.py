# To use this action agent:
# agent = ActionAgent(
#     num_actions_to_generate=5,
#     reflection_threshold=0.7,
#     max_refinement_steps=3
# )
# # The agent will now use GRPO for better decision making
# optimal_action = agent.select_optimal_action(observation)



import json
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import voyager.utils as U
from javascript import require
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from voyager.agents.vision import VisionAgent
from voyager.prompts import load_prompt
from voyager.control_primitives_context import load_control_primitives_context

class ActionAgent:
    def __init__(
        self,
        model_name="gpt-4",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
        vision_agent=None,
        num_actions_to_generate=5,  # Number of candidate actions to generate
        reflection_threshold=0.7,    # Score threshold for action refinement
        max_refinement_steps=3      # Maximum refinement iterations
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        self.num_actions_to_generate = num_actions_to_generate
        self.reflection_threshold = reflection_threshold
        self.max_refinement_steps = max_refinement_steps
        
        # Initialize group context for collaborative optimization
        self.group_context = {
            'other_agents': [],
            'shared_objectives': [],
            'action_history': []
        }
        
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
            # Load additional GRPO-specific state if exists
            try:
                self.group_context = U.load_json(f"{ckpt_dir}/action/group_context.json")
            except:
                pass
        else:
            self.chest_memory = {}

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )

        if vision_agent is None:
            print("\033[33mActionAgent initializing VisionAgent\033[0m")
            self.vision_agent = VisionAgent()
        else:
            self.vision_agent = vision_agent
            
        print("\033[33mActionAgent getting vision_memory\033[0m")
        self.vision_data = self.vision_agent.get_vision_memory()

    def generate_candidate_actions(self, observation: str) -> List[Dict[str, Any]]:
        """Generate multiple candidate actions using the LLM."""
        system_message = self.render_system_message()
        human_message = HumanMessage(content=f"{observation}\n\nGenerate {self.num_actions_to_generate} different possible actions.")
        
        response = self.llm.generate([[system_message, human_message]])
        candidate_actions = []
        
        for generation in response.generations[0]:
            parsed_action = self.process_ai_message(AIMessage(content=generation.text))
            if isinstance(parsed_action, dict):  # Only include valid actions
                candidate_actions.append(parsed_action)
                
        return candidate_actions

    def score_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Score an action based on multiple criteria."""
        # Implement GRPO scoring logic
        scores = {
            'goal_alignment': self._evaluate_goal_alignment(action, context),
            'resource_efficiency': self._evaluate_resource_efficiency(action),
            'safety': self._evaluate_safety(action),
            'group_coordination': self._evaluate_group_coordination(action)
        }
        
        weights = {
            'goal_alignment': 0.4,
            'resource_efficiency': 0.2,
            'safety': 0.2,
            'group_coordination': 0.2
        }
        
        return sum(score * weights[criterion] for criterion, score in scores.items())

    def _evaluate_goal_alignment(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate how well the action aligns with current goals."""
        # Implementation of goal alignment scoring
        system_message = SystemMessage(content="Evaluate how well this action aligns with the current objectives.")
        human_message = HumanMessage(content=f"Action: {action}\nContext: {context}")
        
        response = self.llm.predict_messages([system_message, human_message])
        # Parse response to extract numerical score
        try:
            score = float(re.search(r"Score: ([\d.]+)", response.content).group(1))
            return min(max(score, 0.0), 1.0)  # Normalize to [0,1]
        except:
            return 0.5  # Default score if parsing fails

    def _evaluate_resource_efficiency(self, action: Dict[str, Any]) -> float:
        """Evaluate the resource efficiency of an action."""
        # Analyze resource usage in action code
        code = action.get('program_code', '')
        resource_patterns = {
            'inventory_check': r'bot\.inventory',
            'pathfinding': r'bot\.pathfinder',
            'crafting': r'bot\.craft'
        }
        
        efficiency_score = 1.0
        for pattern in resource_patterns.values():
            if re.search(pattern, code):
                efficiency_score *= 0.9  # Penalize resource usage
                
        return efficiency_score

    def _evaluate_safety(self, action: Dict[str, Any]) -> float:
        """Evaluate the safety of an action."""
        # Check for potentially risky operations
        code = action.get('program_code', '')
        risk_patterns = {
            'height_risk': r'bot\.position\.y',
            'combat_risk': r'bot\.attack',
            'water_risk': r'bot\.swim'
        }
        
        safety_score = 1.0
        for pattern in risk_patterns.values():
            if re.search(pattern, code):
                safety_score *= 0.8  # Penalize risky operations
                
        return safety_score

    def _evaluate_group_coordination(self, action: Dict[str, Any]) -> float:
        """Evaluate how well the action coordinates with other agents."""
        # Check for conflicts with other agents' actions
        conflicts = 0
        for other_action in self.group_context['action_history'][-5:]:  # Look at last 5 actions
            if self._detect_conflict(action, other_action):
                conflicts += 1
                
        return max(0.0, 1.0 - (conflicts * 0.2))

    def _detect_conflict(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Detect if two actions might conflict."""
        # Simple conflict detection based on resource usage and location
        code1 = action1.get('program_code', '')
        code2 = action2.get('program_code', '')
        
        # Check for same resource usage
        resources = ['inventory', 'crafting_table', 'furnace']
        for resource in resources:
            if resource in code1.lower() and resource in code2.lower():
                return True
                
        return False

    def refine_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Refine an action based on self-reflection."""
        system_message = SystemMessage(content="Analyze this action and suggest improvements while maintaining its core objective.")
        human_message = HumanMessage(content=f"Action: {json.dumps(action)}\nContext: {json.dumps(context)}")
        
        response = self.llm.predict_messages([system_message, human_message])
        refined_action = self.process_ai_message(response)
        
        return refined_action if isinstance(refined_action, dict) else action

    def select_optimal_action(self, observation: str) -> Dict[str, Any]:
        """Select the optimal action using GRPO."""
        context = {
            'observation': observation,
            'group_context': self.group_context
        }
        
        # Generate candidate actions
        candidates = self.generate_candidate_actions(observation)
        if not candidates:
            return None

        # Score and refine actions
        scored_actions = []
        for action in candidates:
            score = self.score_action(action, context)
            refined_action = action
            
            # Iteratively refine low-scoring actions
            refinement_steps = 0
            while score < self.reflection_threshold and refinement_steps < self.max_refinement_steps:
                refined_action = self.refine_action(refined_action, context)
                score = self.score_action(refined_action, context)
                refinement_steps += 1
            
            scored_actions.append((refined_action, score))

        # Select the best action
        best_action, _ = max(scored_actions, key=lambda x: x[1])
        
        # Update group context
        self.group_context['action_history'].append(best_action)
        U.dump_json(self.group_context, f"{self.ckpt_dir}/action/group_context.json")
        
        return best_action

    # ... (keeping existing methods like update_chest_memory, render_chest_observation, etc.)
    
    def process_ai_message(self, message: AIMessage) -> Dict[str, Any]:
        """Enhanced version of the original process_ai_message method."""
        if not isinstance(message, AIMessage):
            return f"Error: Expected AIMessage, got {type(message)}"

        retry = 3
        error = None
        while retry > 0:
            try:
                babel = require("@babel/core")
                babel_generator = require("@babel/generator").default

                code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content))
                parsed = babel.parse(code)
                
                functions = []
                assert len(list(parsed.program.body)) > 0, "No functions found"
                
                for node in parsed.program.body:
                    if node.type != "FunctionDeclaration":
                        continue
                    node_type = "AsyncFunctionDeclaration" if node["async"] else "FunctionDeclaration"
                    functions.append({
                        "name": node.id.name,
                        "type": node_type,
                        "body": babel_generator(node).code,
                        "params": list(node["params"]),
                        # Add metadata for GRPO
                        "complexity": len(node.body.body),
                        "resource_usage": self._analyze_resource_usage(node)
                    })

                # Find the last async function
                main_function = None
                for function in reversed(functions):
                    if function["type"] == "AsyncFunctionDeclaration":
                        main_function = function
                        break

                assert main_function is not None, "No async function found. Your main function must be async."
                assert (
                    len(main_function["params"]) == 1 and 
                    main_function["params"][0].name == "bot"
                ), f"Main function {main_function['name']} must take a single argument named 'bot'"

                program_code = "\n\n".join(function["body"] for function in functions)
                exec_code = f"await {main_function['name']}(bot);"
                
                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                    "metadata": {
                        "complexity": main_function["complexity"],
                        "resource_usage": main_function["resource_usage"]
                    }
                }
            except Exception as e:
                retry -= 1
                error = e
                time.sleep(1)
                
        return f"Error parsing action response (before program execution): {error}"

    def _analyze_resource_usage(self, node: Any) -> Dict[str, int]:
        """Analyze the resource usage patterns in a function node."""
        code = require("@babel/generator").default(node).code
        return {
            "inventory_access": len(re.findall(r'bot\.inventory', code)),
            "pathfinding": len(re.findall(r'bot\.pathfinder', code)),
            "crafting": len(re.findall(r'bot\.craft', code)),
            "combat": len(re.findall(r'bot\.attack', code))
        }
    

#     Here are the key improvements:

# Multi-Step Reasoning:
# Added generate_candidate_actions() to create multiple possible actions
# Implemented comprehensive scoring system with score_action() that evaluates:

# Goal alignment
# Resource efficiency
# Safety considerations
# Group coordination

# Self-Reflection & Refinement:
# Added refine_action() for iterative improvement of low-scoring actions
# Implemented reflection threshold and maximum refinement steps
# Enhanced action processing with metadata collection

# Collaborative Group Optimization:
# Added group context tracking
# Implemented conflict detection between actions
# Added action history for coordinated decision making

# Key New Parameters:
# num_actions_to_generate: Number of candidate actions to consider
# reflection_threshold: Score threshold for triggering refinement
# max_refinement_steps: Maximum refinement iterations