import json
import re
import time
import voyager.utils as U
from javascript import require
from langchain.chat_models import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from voyager.agents.vision import VisionAgent
#model_name = ChatAnthropic(model="claude-3.5", temperature=0.7, max_tokens=512)

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
    ):
        # TODO: Add a parameter to the constructor for the Graph RAG approach
        # self.scene_graph = scene_graph()  # Initialize the scene graph in the action agent
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
            request_timeout=request_timout,
        )

        # TODO: Modify the ActionAgent to use vision agent's insights for reasoning about actions.
        # Improve navigation and interaction in ActionAgent.
         # Use the passed vision_agent if provided, otherwise create a new instance
        if vision_agent is None:
            print("\033[33mActionAgent initializing VisionAgent\033[0m")
            self.vision_agent = VisionAgent()  # Create a new instance if none is provided
        else:
            self.vision_agent = vision_agent  # Use the provided instance
        print("\033[33mActionAgent getting vision_memory\033[0m")
        vision_data = self.vision_agent.get_vision_memory()

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
        # if self.vision_agent:
        #     visual_context = self.vision_agent.capture_and_analyze("path_to_image.png")
        #     observation += f"Visual Analysis:\n{visual_context}\n\n"
        # Fetch the latest vision data
        vision_data = self.vision_agent.get_vision_memory()
        formatted_vision_data = json.dumps(vision_data, indent=2)

        # Incorporate vision data into the observation
        observation += f"Vision Data:\n{formatted_vision_data}\n\n"

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
    
    # TODO: Implement the process_action method
    # def process_action(self, events):
    #     # Query the scene graph to get nearby objects
    #     nearby_objects = self.scene_graph.query(current_state["object_id"], relation="nearby")
    #     # Use the nearby objects and other relationships for reasoning
    #     pass

    # def analyze_vision_data(self, vision_data):
    #     """
    #     Parse vision data to identify resources and prioritize tasks.
    #     """
    #     resource_map = {
    #         "tree": "Harvest wood",
    #         "exposed_stone": "Mine cobblestone",
    #         "water": "Collect water bucket",
    #         "vegetation": "Gather seeds or food",
    #     }
    #     tasks = [resource_map[resource] for resource in vision_data if resource in resource_map]
    #     return tasks
    def analyze_vision_data(self, vision_data):
        """
        Parse vision data to identify resources and prioritize tasks.
        """
        tasks = []

        for timestamp, data in vision_data.items():
            # Analyze optimal_block
            optimal_block = data.get("optimal_block", {})
            block_type = optimal_block.get("type", "")
            if block_type.endswith("_log"):
                tasks.append("Harvest wood")

            # Analyze other_blocks
            for block in data.get("other_blocks", []):
                block_type = block.get("type", "")
                if block_type == "exposed_stone":
                    tasks.append("Mine cobblestone")
                elif block_type == "water":
                    tasks.append("Collect water bucket")
                elif block_type in ["mangrove_leaves", "moss_carpet", "vine"]:
                    tasks.append("Gather seeds or food")
                # Add more conditions as needed

        # Remove duplicates
        tasks = list(set(tasks))

        return tasks

    # def guide_agent_actions(self, vision_data, inventory_status):
    #     """
    #     Generate a list of prioritized actions for the agent based on vision data.
    #     """
    #     tasks = self.analyze_vision_data(vision_data)
    #     prioritized_tasks = []

    #     # Check inventory constraints
    #     if inventory_status["space"] < len(tasks):
    #         tasks.append("Deposit items into chest")

    #     # Prioritize tasks
    #     for task in tasks:
    #         if "Harvest wood" in task and inventory_status["tools"]["axe"]:
    #             prioritized_tasks.append(task)
    #         elif "Mine cobblestone" in task and inventory_status["tools"]["pickaxe"]:
    #             prioritized_tasks.append(task)
    #         else:
    #             prioritized_tasks.append("Craft necessary tools")

    #     return prioritized_tasks
    def guide_agent_actions(self, vision_data, inventory_status):
        """
        Generate a list of prioritized actions for the agent based on vision data.
        """
        tasks = self.analyze_vision_data(vision_data)
        prioritized_tasks = []

        # Check inventory constraints
        if inventory_status.get("space", 0) < len(tasks):
            tasks.append("Deposit items into chest")

        # Prioritize tasks based on available tools
        for task in tasks:
            if "Harvest wood" in task and inventory_status.get("tools", {}).get("axe", False):
                prioritized_tasks.append(task)
            elif "Mine cobblestone" in task and inventory_status.get("tools", {}).get("pickaxe", False):
                prioritized_tasks.append(task)
            elif "Collect water bucket" in task and inventory_status.get("tools", {}).get("bucket", False):
                prioritized_tasks.append(task)
            else:
                prioritized_tasks.append("Craft necessary tools")

        # Remove duplicates while preserving order
        seen = set()
        prioritized_tasks = [x for x in prioritized_tasks if not (x in seen or seen.add(x))]

        return prioritized_tasks


# Update Vision Data Regularly: Instead of fetching vision data only during initialization, ensure that the ActionAgent retrieves the latest vision data whenever it needs to make a decision. This can be achieved by calling self.vision_agent.get_vision_memory() at appropriate points in the ActionAgent's workflow.