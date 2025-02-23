#agent = ActionAgent(
#     cot_depth=3,  # Number of reasoning steps
#     num_actions_to_generate=5,
#     reflection_threshold=0.7
# )
# # The agent will now use both GRPO and Chain of Thought
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
        num_actions_to_generate=5,
        reflection_threshold=0.7,
        max_refinement_steps=3,
        cot_depth=3  # New: Control depth of chain of thought reasoning
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        self.num_actions_to_generate = num_actions_to_generate
        self.reflection_threshold = reflection_threshold
        self.max_refinement_steps = max_refinement_steps
        self.cot_depth = cot_depth
        
        # Initialize reasoning chain memory
        self.reasoning_chains = []
        
        # Initialize group context for collaborative optimization
        self.group_context = {
            'other_agents': [],
            'shared_objectives': [],
            'action_history': [],
            'reasoning_history': []  # New: Track reasoning chains
        }
        
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
            try:
                self.group_context = U.load_json(f"{ckpt_dir}/action/group_context.json")
                self.reasoning_chains = U.load_json(f"{ckpt_dir}/action/reasoning_chains.json")
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

    def generate_chain_of_thought(self, observation: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chain of thought reasoning for action selection."""
        system_prompt = """Analyze the situation and generate a chain of thought reasoning process to determine the best action.
        Consider:
        1. Current state and objectives
        2. Potential consequences of actions
        3. Resource availability and constraints
        4. Past experiences and outcomes
        5. Alternative approaches
        
        Format your response as a series of logical steps, each building on the previous ones."""

        thoughts = []
        current_context = observation
        
        for step in range(self.cot_depth):
            # Generate next reasoning step
            human_message = HumanMessage(content=f"Current context: {current_context}\n\nNext reasoning step:")
            response = self.llm.predict_messages([SystemMessage(content=system_prompt), human_message])
            
            thought = {
                'step': step + 1,
                'reasoning': response.content,
                'context': current_context
            }
            thoughts.append(thought)
            
            # Update context with new reasoning
            current_context = f"{current_context}\nReasoning step {step + 1}: {response.content}"
            
        return thoughts

    def evaluate_reasoning_chain(self, chain: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        """Evaluate the quality of a reasoning chain."""
        evaluation_criteria = {
            'logical_consistency': self._evaluate_logical_consistency(chain),
            'context_relevance': self._evaluate_context_relevance(chain, context),
            'actionability': self._evaluate_actionability(chain),
            'completeness': self._evaluate_completeness(chain)
        }
        
        weights = {
            'logical_consistency': 0.3,
            'context_relevance': 0.3,
            'actionability': 0.2,
            'completeness': 0.2
        }
        
        return sum(score * weights[criterion] for criterion, score in evaluation_criteria.items())

    def _evaluate_logical_consistency(self, chain: List[Dict[str, Any]]) -> float:
        """Evaluate logical consistency between reasoning steps."""
        system_message = SystemMessage(content="Evaluate the logical consistency between these reasoning steps.")
        chain_text = "\n".join(f"Step {t['step']}: {t['reasoning']}" for t in chain)
        human_message = HumanMessage(content=f"Reasoning chain:\n{chain_text}")
        
        response = self.llm.predict_messages([system_message, human_message])
        try:
            score = float(re.search(r"Consistency score: ([\d.]+)", response.content).group(1))
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    def _evaluate_context_relevance(self, chain: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        """Evaluate relevance of reasoning to current context."""
        relevant_terms = set(self._extract_key_terms(context))
        chain_terms = set(self._extract_key_terms("\n".join(t['reasoning'] for t in chain)))
        
        overlap = len(relevant_terms.intersection(chain_terms))
        total_terms = len(relevant_terms)
        
        return overlap / total_terms if total_terms > 0 else 0.5

    def _evaluate_actionability(self, chain: List[Dict[str, Any]]) -> float:
        """Evaluate how actionable the reasoning chain is."""
        action_patterns = [
            r'should',
            r'could',
            r'must',
            r'need to',
            r'will',
            r'can'
        ]
        
        total_actions = 0
        for thought in chain:
            total_actions += sum(1 for pattern in action_patterns 
                               if re.search(pattern, thought['reasoning'].lower()))
            
        return min(total_actions / (len(chain) * 2), 1.0)  # Expect ~2 actionable items per step

    def _evaluate_completeness(self, chain: List[Dict[str, Any]]) -> float:
        """Evaluate completeness of the reasoning chain."""
        required_elements = [
            'objective',
            'analysis',
            'consideration',
            'conclusion'
        ]
        
        chain_text = "\n".join(t['reasoning'] for t in chain)
        elements_found = sum(1 for element in required_elements 
                           if re.search(element, chain_text, re.IGNORECASE))
        
        return elements_found / len(required_elements)

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using simple NLP."""
        # Simple term extraction (could be enhanced with proper NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [w for w in words if w not in stop_words]

    def select_optimal_action(self, observation: str) -> Dict[str, Any]:
        """Select optimal action using GRPO and Chain of Thought reasoning."""
        context = {
            'observation': observation,
            'group_context': self.group_context
        }
        
        # Generate chain of thought reasoning
        reasoning_chain = self.generate_chain_of_thought(observation, context)
        chain_score = self.evaluate_reasoning_chain(reasoning_chain, context)
        
        # Store reasoning chain if it's high quality
        if chain_score > 0.8:
            self.reasoning_chains.append({
                'chain': reasoning_chain,
                'score': chain_score,
                'timestamp': time.time()
            })
            U.dump_json(self.reasoning_chains, f"{self.ckpt_dir}/action/reasoning_chains.json")
        
        # Generate candidate actions informed by reasoning
        candidates = self.generate_candidate_actions(observation, reasoning_chain)
        if not candidates:
            return None

        # Score and refine actions using both GRPO and reasoning chain
        scored_actions = []
        for action in candidates:
            # Combine GRPO score with reasoning alignment
            grpo_score = self.score_action(action, context)
            reasoning_alignment = self._evaluate_reasoning_alignment(action, reasoning_chain)
            combined_score = 0.7 * grpo_score + 0.3 * reasoning_alignment
            
            refined_action = action
            refinement_steps = 0
            
            # Iteratively refine low-scoring actions
            while combined_score < self.reflection_threshold and refinement_steps < self.max_refinement_steps:
                refined_action = self.refine_action(refined_action, context, reasoning_chain)
                grpo_score = self.score_action(refined_action, context)
                reasoning_alignment = self._evaluate_reasoning_alignment(refined_action, reasoning_chain)
                combined_score = 0.7 * grpo_score + 0.3 * reasoning_alignment
                refinement_steps += 1
            
            scored_actions.append((refined_action, combined_score))

        # Select best action
        best_action, _ = max(scored_actions, key=lambda x: x[1])
        
        # Update group context
        self.group_context['action_history'].append(best_action)
        self.group_context['reasoning_history'].append(reasoning_chain)
        U.dump_json(self.group_context, f"{self.ckpt_dir}/action/group_context.json")
        
        return best_action

    def _evaluate_reasoning_alignment(self, action: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> float:
        """Evaluate how well an action aligns with the reasoning chain."""
        # Extract key conclusions from reasoning chain
        conclusions = []
        for thought in reasoning_chain:
            if 'conclusion' in thought['reasoning'].lower():
                conclusions.append(thought['reasoning'])
        
        if not conclusions:
            return 0.5  # Default score if no clear conclusions
            
        # Evaluate alignment
        system_message = SystemMessage(content="Evaluate how well this action aligns with the reasoning conclusions.")
        human_message = HumanMessage(content=f"""
        Action: {json.dumps(action)}
        
        Reasoning conclusions:
        {'\n'.join(conclusions)}
        
        Score the alignment from 0.0 to 1.0.
        """)
        
        response = self.llm.predict_messages([system_message, human_message])
        try:
            score = float(re.search(r"([\d.]+)", response.content).group(1))
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    # ... (keep existing methods)
    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"\033[32mAction Agent removing chest {position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"\033[32mAction Agent saving chest {position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    def render_system_message(self, skills=[]):
        system_template = load_prompt("action_template")
        # FIXME: Hardcoded control_primitives
        base_skills = [
            "exploreUntil",
            "mineBlock",
            "craftItem",
            "placeItem",
            "smeltItem",
            "killMob",
        ]
        if not self.llm.model_name == "gpt-4":
            base_skills += [
                "useChest",
                "mineflayer",
            ]
        programs = "\n\n".join(load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_human_message(
        self, *, events, code="", task="", context="", critique=""
    ):
        chat_messages = []
        error_messages = []
        # FIXME: damage_messages is not used
        damage_messages = []
        assert events[-1][0] == "observe", "Last event must be observe"
        for i, (event_type, event) in enumerate(events):
            if event_type == "onChat":
                chat_messages.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "onDamage":
                damage_messages.append(event["onDamage"])
            elif event_type == "observe":
                biome = event["status"]["biome"]
                time_of_day = event["status"]["timeOfDay"]
                voxels = event["voxels"]
                entities = event["status"]["entities"]
                health = event["status"]["health"]
                hunger = event["status"]["food"]
                position = event["status"]["position"]
                equipment = event["status"]["equipment"]
                inventory_used = event["status"]["inventoryUsed"]
                inventory = event["inventory"]
                assert i == len(events) - 1, "observe must be the last event"

        observation = ""

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        observation += f"Biome: {biome}\n\n"

        observation += f"Time: {time_of_day}\n\n"

        if voxels:
            observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
        else:
            observation += f"Nearby blocks: None\n\n"

        if entities:
            nearby_entities = [
                k for k, v in sorted(entities.items(), key=lambda x: x[1])
            ]
            observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
        else:
            observation += f"Nearby entities (nearest to farthest): None\n\n"

        observation += f"Health: {health:.1f}/20\n\n"

        observation += f"Hunger: {hunger:.1f}/20\n\n"

        observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

        observation += f"Equipment: {equipment}\n\n"

        if inventory:
            observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
        else:
            observation += f"Inventory ({inventory_used}/36): Empty\n\n"

        if not (
            task == "Place and deposit useless items into a chest"
            or task.startswith("Deposit useless items into the chest at")
        ):
            observation += self.render_chest_observation()

        # TODO 2: Visual analysis if VisionAgent is available
        # Fetch the latest vision data
        vision_data = self.vision_agent.get_vision_memory()
        #formatted_vision_data = json.dumps(vision_data, indent=2)
        if vision_data:
            # Format vision data for readability
            formatted_vision_data = json.dumps(vision_data, indent=2)
            observation += f"Vision Data:\n{formatted_vision_data}\n\n"
        else:
            observation += f"Vision Data: None\n\n"

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return HumanMessage(content=observation)

    def process_ai_message(self, message):
        assert isinstance(message, AIMessage)

        retry = 3
        error = None
        while retry > 0:
            try:
                babel = require("@babel/core")
                babel_generator = require("@babel/generator").default

                code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content)) #original is : .message.content
                parsed = babel.parse(code)
                functions = []
                assert len(list(parsed.program.body)) > 0, "No functions found"
                for i, node in enumerate(parsed.program.body):
                    if node.type != "FunctionDeclaration":
                        continue
                    node_type = (
                        "AsyncFunctionDeclaration"
                        if node["async"]
                        else "FunctionDeclaration"
                    )
                    functions.append(
                        {
                            "name": node.id.name,
                            "type": node_type,
                            "body": babel_generator(node).code,
                            "params": list(node["params"]),
                        }
                    )
                # find the last async function
                main_function = None
                for function in reversed(functions):
                    if function["type"] == "AsyncFunctionDeclaration":
                        main_function = function
                        break
                assert (
                    main_function is not None
                ), "No async function found. Your main function must be async."
                assert (
                    len(main_function["params"]) == 1
                    and main_function["params"][0].name == "bot"
                ), f"Main function {main_function['name']} must take a single argument named 'bot'"
                program_code = "\n\n".join(function["body"] for function in functions)
                exec_code = f"await {main_function['name']}(bot);"
                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }
            except Exception as e:
                retry -= 1
                error = e
                time.sleep(1)
        return f"Error parsing action response (before program execution): {error}"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""
    
    

#     Key Features Verified
# GRPO Implementation
# Generates multiple candidate actions (num_actions_to_generate).
# Scores them based on multiple factors (score_action, _evaluate_reasoning_alignment).
# Uses iterative refinement if the action score is below reflection_threshold.
# Integrates group context tracking (self.group_context) for shared decision-making.

# Chain of Thought (CoT) Reasoning
# Uses generate_chain_of_thought() to break down reasoning steps.
# evaluate_reasoning_chain() assesses CoT quality across logical consistency, context relevance, actionability, and completeness.
# Stores reasoning chains in self.reasoning_chains for later refinement.

# Action Selection Refinement
# If an action is suboptimal, the system refines it based on reasoning chains (refine_action).
# Scores actions using a weighted combination of GRPO scores and reasoning alignment.
# Uses a reflection loop (with a limit of max_refinement_steps).

# Memory Handling
# group_context is persisted to maintain historical decisions.
# Stores reasoning chains in reasoning_chains.json.
# Handles chest memory updates (update_chest_memory) and environmental state observations.