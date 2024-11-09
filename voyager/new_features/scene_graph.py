"""
Since the scene graph is a representation of the relationships between objects in the environment, 
we'll need to implement this data structure and logic for reasoning in your agent."""

"""Implement functions for creating, updating, and querying the scene graph.
"""
# TODO 1: How extend this logic to integrate spatial relationships between objects for planning tasks.

# Implement the SceneGraph class
class SceneGraph:
    def __init__(self):
        self.graph = {}

    def add_object(self, obj_id, obj_type):
        if obj_id not in self.graph:
            self.graph[obj_id] = {"type": obj_type, "relations": []}

    def add_relation(self, obj_id_1, relation, obj_id_2):
        if obj_id_1 in self.graph and obj_id_2 in self.graph:
            self.graph[obj_id_1]["relations"].append((relation, obj_id_2))

    def query(self, obj_id, relation=None):
        if obj_id in self.graph:
            if relation:
                return [obj for rel, obj in self.graph[obj_id]["relations"] if rel == relation]
            return self.graph[obj_id]["relations"]
