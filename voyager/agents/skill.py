# GRPO Enhancement:
# The Skill Agent learns from group-based reward comparisons, selecting the most reusable skills instead of just optimizing for immediate success.
# Example: If an agent learns both "building a bridge" and "swimming across rivers", GRPO allows it to determine which is more generally useful across tasks.

import os

import voyager.utils as U
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma

from voyager.prompts import load_prompt
from voyager.control_primitives import load_control_primitives
#from voyager.agents.vision import VisionAgent
# TODO 1: from utils.scene_graph import SceneGraph  # Import the SceneGraph if not already done
# TODO 1: Create a new class within the file that uses the Graph RAG approach to retrieve relevant skills based on graph embeddings or scene graph queries.
# from dspy.graph import Graph
# from dspy.graph import Node, Edge
# from dspy.graph import GraphRAG
# from dspy.graph import GraphRAGNode, GraphRAGEdge

# TODO 2: add dspy prompt only first
import dspy
#from dspy import Prompt, Program, Signature, Value # Import the necessary classes from the dspy module
#from dspy.skill_manager import DspySkillManager
# @dataclass
# class MineBlock(Program):
#     block_type: Value[str] = Value(default="stone")
#     tool: Value[str] = Value(default="wooden_pickaxe")
#     def __call__(self, block_type: str, tool: str) -> str:
#         # Code to execute mining action using Minecraft API
#         # ...
#         return f"Mined {block_type} with {tool}"
# # TODO 2: Define other skills similarly...   
# class CraftPlanks(Program):
#     pass
# """
# end of added code for TODO 2
# """

class SkillManager:
    def __init__(
        self,
        model_name="gpt-4",
        # model_name= dspy,
        temperature=0,
        retrieval_top_k=5,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        #vision_agent: VisionAgent | None = None,
        vision_agent=None,
    ):
        #TODO 1: Add a parameter to the constructor for the Graph RAG approach
        # self.scene_graph = scene_graph()  # Initialize the scene graph in the skill manager
        #TODO 1: Add a parameter to the constructor for the Graph RAG approach
        # self.graph_rag_manager = GraphRAGManager()  # Initialize the Graph RAG manager in the skill manager
        #TODO 2: Add a parameter to the constructor for the dspy skill manager
        # # self.dspy_skill_manager = DspySkillManager()  # Initialize the dspy skill manager in the skill manager
        # self.dspy_skills = [MineBlock, CraftPlanks, ...]  # Store DSPy skills in the skill manager

        # # def some_method(self, ...):
        # #    # ... access and utilize DSPy skills ...
        # #    skill = self.dspy_skills[0]  # Get the MineBlock skill
        # #    result = skill(block_type="dirt", tool="iron_shovel")  # Execute skill with parameters
        
        # def select_skill(self, observation):
        #     """
        #     Purpose: This method would be responsible for selecting the most appropriate DSPy skill based on the current observation of the Minecraft environment.
        #     Implementation: It could utilize various techniques, such as rule-based logic, machine learning models, 
        #     or even prompting an LLM to determine the best skill for the given situation.
        #     # Analyze observation and select a skill from self.dspy_skills
        #     # ...
        #     return selected_skill"""  # Returns a DSPy skill object
        
        # def execute_skill(self, skill, observation):
        #     """
        #     Purpose: This method would be responsible for executing the selected DSPy skill based on the current observation of the Minecraft environment.
        #     Implementation: It would call the selected skill object with the appropriate parameters based on the observation.
        #     # Execute the selected skill with the observation
        #     def execute_skill(self, skill, **kwargs):
        #     result = skill(**kwargs)  # Execute the skill
        #     # ... handle result and update state ...
        #     return result"""

        def generate_prompt_for_skill(self, skill, observation):
            """
            Purpose: This method would generate a prompt for an LLM, incorporating information about the selected skill and the current observation.
            Implementation: It could use DSPy's declarative language to create dynamic prompt templates, similar to the example you provided earlier.
            
            Create prompt templates using DSPy's declarative language. These templates will serve as blueprints for generating prompts, allowing you to incorporate dynamic information and conditions.
            """
            prompt_template = """
            Task: {task_description}
            Observation: {observation}
            Skill: {skill_name}
            """
            prompt = self.dspy_agent.generate_prompt(
                prompt_template,
                task_description="Mine a block",
                observation=observation,
                skill_name=skill.__name__,
            )
            return prompt
        
        # def update_skill_library(self, new_skill):
        #     """
        #     Purpose: This method would add new DSPy skills to the SkillManager's library.
        #     Implementation: It would append the new skill to the self.dspy_skills list.
        #     """
        #     self.dspy_skills.append(new_skill)
        
        # def get_skill_by_name(self, skill_name):
        #     """
        #     Purpose: This method would retrieve a DSPy skill from the library based on its name.
        #     Implementation: It could iterate through self.dspy_skills and return the skill with the matching name.
        #     """
        #     for skill in self.dspy_skills:
        #         if skill.__name__ == skill_name:
        #             return skill
        #     return None  # Skill not found
        # """
        # end of added code for TODO 2
        # """
        self.vision_agent = vision_agent

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )
        U.f_mkdir(f"{ckpt_dir}/skill/code")
        U.f_mkdir(f"{ckpt_dir}/skill/description")
        U.f_mkdir(f"{ckpt_dir}/skill/vectordb")
        # programs for env execution
        self.control_primitives = load_control_primitives()
        if resume:
            print(f"\033[33mLoading Skill Manager from {ckpt_dir}/skill\033[0m")
            self.skills = U.load_json(f"{ckpt_dir}/skill/skills.json")
        else:
            self.skills = {}
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/skill/vectordb",
        )
        assert self.vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb is not synced with skills.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but {len(self.skills)} skills in skills.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    @property
    def programs(self):
        programs = ""
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.control_primitives:
            programs += f"{primitives}\n\n"
        return programs

    def add_new_skill(self, info):
        if info["task"].startswith("Deposit useless items into the chest at"):
            # No need to reuse the deposit skill
            return
        program_name = info["program_name"]
        program_code = info["program_code"]
        skill_description = self.generate_skill_description(program_name, program_code)
        print(
            f"\033[33mSkill Manager generated description for {program_name}:\n{skill_description}\033[0m"
        )
        if program_name in self.skills:
            print(f"\033[33mSkill {program_name} already exists. Rewriting!\033[0m")
            self.vectordb._collection.delete(ids=[program_name])
            i = 2
            while f"{program_name}V{i}.js" in os.listdir(f"{self.ckpt_dir}/skill/code"):
                i += 1
            dumped_program_name = f"{program_name}V{i}"
        else:
            dumped_program_name = program_name
        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        self.skills[program_name] = {
            "code": program_code,
            "description": skill_description,
        }
        assert self.vectordb._collection.count() == len(
            self.skills
        ), "vectordb is not synced with skills.json"
        U.dump_text(
            program_code, f"{self.ckpt_dir}/skill/code/{dumped_program_name}.js"
        )
        U.dump_text(
            skill_description,
            f"{self.ckpt_dir}/skill/description/{dumped_program_name}.txt",
        )
        U.dump_json(self.skills, f"{self.ckpt_dir}/skill/skills.json")
        self.vectordb.persist()

    def generate_skill_description(self, program_name, program_code):
        messages = [
            SystemMessage(content=load_prompt("skill")),
            HumanMessage(
                content=program_code
                + "\n\n"
                + f"The main function is `{program_name}`."
            ),
        ]
        skill_description = f"    // { self.llm(messages).content}"
        return f"async function {program_name}(bot) {{\n{skill_description}\n}}"

    # Implement a function that retrieves relevant skills based on the query using the vector database.
    def retrieve_skills(self, query):
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        print(f"\033[33mSkill Manager retrieving for {k} skills\033[0m")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        print(
            f"\033[33mSkill Manager retrieved skills: "
            f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}\033[0m"
        )
        skills = []
        for doc, _ in docs_and_scores:
            skills.append(self.skills[doc.metadata["name"]]["code"])
        return skills




"""
TODO 1: Implement a new class that uses the Graph RAG approach to retrieve relevant skills based on graph embeddings or scene graph queries.
"""
    # TODO 1: Modify retrieve_skills to Incorporate Scene Graph Queries
    # def retrieve_skills(self, query, current_state):
    #     # First, query the scene graph based on the current state
    #     nearby_objects = self.scene_graph.query(current_state["object_id"], relation="nearby")
        
    #     # Use nearby objects or relationships from the scene graph to enhance the query
    #     enhanced_query = f"{query} with nearby objects: {nearby_objects}"
        
    #     # Continue with the existing skill retrieval mechanism
    #     k = min(self.vectordb._collection.count(), self.retrieval_top_k)
    #     if k == 0:
    #         return []
        
    #     print(f"\033[33mSkill Manager retrieving for {k} skills based on enhanced query\033[0m")
        
    #     # Use the enhanced query that includes context from the scene graph
    #     docs_and_scores = self.vectordb.similarity_search_with_score(enhanced_query, k=k)
        
    #     print(
    #         f"\033[33mSkill Manager retrieved skills: "
    #         f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}\033[0m"
    #     )
        
    #     skills = []
    #     for doc, _ in docs_and_scores:
    #         skills.append(self.skills[doc.metadata["name"]]["code"])
    
    #     return skills
    # TODO 1: Implement a function that uses the Graph RAG approach to retrieve relevant skills based on graph embeddings or scene graph queries.
    # def retrieve_skills_with_rag(self, query, current_state):
    #     # Query the scene graph for context
    #     nearby_objects = self.scene_graph.query(current_state["object_id"], relation="nearby")
        
    #     # Retrieve relevant nodes or subgraphs using Graph RAG
    #     graph_rag_retrieved_nodes = self.graph_rag_retrieve(nearby_objects)
        
    #     # Enhance the query with both scene graph nodes and retrieved graph context
    #     enhanced_query = f"{query} with nearby objects: {nearby_objects} and graph context: {graph_rag_retrieved_nodes}"
        
    #     # Continue with skill retrieval as before
    #     k = min(self.vectordb._collection.count(), self.retrieval_top_k)
    #     if k == 0:
    #         return []
        
    #     print(f"\033[33mSkill Manager retrieving for {k} skills based on enhanced query with Graph RAG\033[0m")
    #     docs_and_scores = self.vectordb.similarity_search_with_score(enhanced_query, k=k)
        
    #     print(
    #         f"\033[33mSkill Manager retrieved skills: "
    #         f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}\033[0m"
    #     )
        
    #     skills = []
    #     for doc, _ in docs_and_scores:
    #         skills.append(self.skills[doc.metadata["name"]]["code"])
        
    #     return skills

    # TODO 1: Add your graph_rag_retrieve method here
    # def graph_rag_retrieve(self, nearby_objects):
    #     # Example logic for retrieving relevant nodes from the scene graph
    #     # You can embed the graph nodes using a graph neural network or a similar model
    #     related_nodes = []
    #     for obj in nearby_objects:
    #         related_nodes.extend(self.scene_graph.query(obj, relation="connected_to"))
    #     return related_nodes
