from voyager.prompts import load_prompt
from voyager.utils.json_utils import fix_and_parse_json
from langchain.chat_models import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from voyager.agents.vision import VisionAgent
import json
import time

# TODO: Create a new class within the file that uses the Graph RAG approach to retrieve relevant skills based on graph embeddings or scene graph queries.
class CriticAgent:
    def __init__(
        self,
        model_name="gpt-4",
        temperature=0,
        request_timout=120,
        mode="auto",
        vision_agent=None,
    ):

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )

        assert mode in ["auto", "manual"]
        self.mode = mode

         # TODO: Modify the ActionAgent to use vision agent's insights for reasoning about actions.
        # Evaluate task success in CriticAgent.
        self.vision_agent = vision_agent
        vision_data = self.vision_agent.get_vision_memory() 

    def render_system_message(self):
        system_message = SystemMessage(content=load_prompt("critic"))
        return system_message

    def render_human_message(self, *, events, task, context, chest_observation):
        """
        Constructs a human-readable observation message for the LLM.

        Parameters:
            events (list): A list of tuples representing events.
            code (str): The code from the last round of actions.
            task (str): The current task assigned to the bot.
            context (str): Additional context or instructions.
            critique (str): Feedback or critique from previous evaluations.

        Returns:
            HumanMessage: The constructed human message containing observations.
        """
        assert events[-1][0] == "observe", "Last event must be observe"
        biome = events[-1][1]["status"]["biome"]
        time_of_day = events[-1][1]["status"]["timeOfDay"]
        voxels = events[-1][1]["voxels"]
        health = events[-1][1]["status"]["health"]
        hunger = events[-1][1]["status"]["food"]
        position = events[-1][1]["status"]["position"]
        equipment = events[-1][1]["status"]["equipment"]
        inventory_used = events[-1][1]["status"]["inventoryUsed"]
        inventory = events[-1][1]["inventory"]

        for i, (event_type, event) in enumerate(events):
            if event_type == "onError":
                print(f"\033[31mCritic Agent: Error occurs {event['onError']}\033[0m")
                return None

        observation = ""

        observation += f"Biome: {biome}\n\n"

        observation += f"Time: {time_of_day}\n\n"

        if voxels:
            observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
        else:
            observation += f"Nearby blocks: None\n\n"

        observation += f"Health: {health:.1f}/20\n\n"
        observation += f"Hunger: {hunger:.1f}/20\n\n"

        observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

        observation += f"Equipment: {equipment}\n\n"

        if inventory:
            observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
        else:
            observation += f"Inventory ({inventory_used}/36): Empty\n\n"

        observation += chest_observation

         # Vision Data Integration
        vision_data = self.vision_agent.get_vision_memory()
        vision_description = self.format_vision_data(vision_data)
        observation += f"Vision Insights:\n{vision_description}\n\n"

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        print(f"\033[31m****Critic Agent human message****\n{observation}\033[0m")
        return HumanMessage(content=observation)

    def human_check_task_success(self):
        """
        Allows manual input to confirm task success.

        Returns:
            tuple: (success (bool), critique (str))
        """
        confirmed = False
        success = False
        critique = ""
        while not confirmed:
            success = input("Success? (y/n)")
            success = success.lower() == "y"
            critique = input("Enter your critique:")
            print(f"Success: {success}\nCritique: {critique}")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        return success, critique

    def ai_check_task_success(self, messages, max_retries=5):
        """
        Utilizes the LLM to automatically assess task success.

        Parameters:
            messages (list): A list containing system and human messages.
            max_retries (int): Maximum number of retries for parsing.

        Returns:
            tuple: (success (bool), critique (str))
        """
        if max_retries == 0:
            print(
                "\033[31mFailed to parse Critic Agent response. Consider updating your prompt.\033[0m"
            )
            return False, ""

        if messages[1] is None:
            return False, ""

        critic = self.llm(messages).content
        print(f"\033[31m****Critic Agent ai message****\n{critic}\033[0m")
        try:
            response = fix_and_parse_json(critic)
            assert response["success"] in [True, False]
            if "critique" not in response:
                response["critique"] = ""
            return response["success"], response["critique"]
        except Exception as e:
            print(f"\033[31mError parsing critic response: {e} Trying again!\033[0m")
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def check_task_success(
        self, *, events, task, context, chest_observation, max_retries=5
    ):
        human_message = self.render_human_message(
            events=events,
            task=task,
            context=context,
            chest_observation=chest_observation,
        )

        messages = [
            self.render_system_message(),
            human_message,
        ]

        if self.mode == "manual":
            return self.human_check_task_success()
        elif self.mode == "auto":
            return self.ai_check_task_success(
                messages=messages, max_retries=max_retries
            )
        else:
            raise ValueError(f"Invalid critic agent mode: {self.mode}")

    def format_vision_data(self, vision_data):
        """
        Converts vision data into descriptive insights.

        Parameters:
            vision_data (dict): The raw vision memory data.

        Returns:
            str: A descriptive summary of vision insights.
        """
        vision_insights = []
        for timestamp, data in vision_data.items():
            # Process optimal_block
            optimal_block = data.get("optimal_block", {})
            if optimal_block:
                block_type = optimal_block.get("type", "Unknown")
                position = optimal_block.get("position", {})
                accessibility = optimal_block.get("accessibility", False)
                vision_insights.append(
                    f"At {timestamp}, detected a {block_type} at coordinates "
                    f"({position.get('x', 0)}, {position.get('y', 0)}, {position.get('z', 0)}) "
                    f"which is {'accessible' if accessibility else 'not accessible'}."
                )
            # Process other_blocks
            for block in data.get("other_blocks", []):
                block_type = block.get("type", "Unknown")
                position = block.get("position", {})
                accessibility = block.get("accessibility", False)
                vision_insights.append(
                    f"Detected a {block_type} at coordinates "
                    f"({position.get('x', 0)}, {position.get('y', 0)}, {position.get('z', 0)}) "
                    f"which is {'accessible' if accessibility else 'not accessible'}."
                )
        
        return "\n".join(vision_insights) if vision_insights else "None"