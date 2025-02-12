# __init__.py in the reasoning folder

from .graphrag import GraphRAGManager
from .scene_graph import SceneGraph
from .instruction_learning import InstructionLearner

# Initialize reasoning-related classes for Voyager integration
class ReasoningManager:
    def __init__(self, graph_db=None, scene_graph_data=None):
        """
        Initializes the reasoning manager with components for Graph RAG, scene graph-based reasoning, 
        and instruction-based learning for complex decision making.
        
        :param graph_db: Database or vector store for Graph RAG queries
        :param scene_graph_data: Initial data for constructing the scene graph
        """
        # GraphRAG for reasoning with external knowledge and retrieval
        self.graph_rag = GraphRAGManager(graph_db) if graph_db else None

        # Scene Graph for spatial and relational understanding
        self.scene_graph = SceneGraph(scene_graph_data) if scene_graph_data else None

        # Instruction Learner for parsing and executing structured instructions
        self.instruction_learner = InstructionLearner()

    def perform_reasoning(self, task_description):
        """
        Example method to demonstrate how reasoning components might be called to handle a task.
        :param task_description: A description of the task for the reasoning modules to process.
        :return: Result from reasoning, potentially including context, related knowledge, and planned actions.
        """
        # Step 1: Parse task description using instruction learner
        instructions = self.instruction_learner.parse_instruction(task_description)

        # Step 2: Query relevant information from GraphRAG
        related_knowledge = self.graph_rag.retrieve_with_graph(instructions) if self.graph_rag else None

        # Step 3: Use the scene graph for context-aware decision making
        scene_context = self.scene_graph.query_scene(instructions) if self.scene_graph else None

        return {
            "instructions": instructions,
            "related_knowledge": related_knowledge,
            "scene_context": scene_context,
        }

# Export core classes for ease of access
__all__ = ["GraphRAGManager", "SceneGraph", "InstructionLearner", "ReasoningManager"]
