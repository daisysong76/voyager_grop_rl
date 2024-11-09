# TODO 3: Abstracting API Calls Using Adapters
# Instead of handling each API (OpenAI, Anthropic) within VisionAgent, use an adapter pattern to encapsulate the differences between APIs. 
# This way, each API call is defined in its own class, making it easier to add new vision APIs or swap out existing ones without modifying VisionAgent.

import openai
import anthropic

class VisionAgent:
    def __init__(self, object_detector, depth_estimator=None, multimodal_model=None):
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.multimodal_model = multimodal_model

    def analyze_image(self, image_path):
        objects = self.detect_objects(image_path)
        spatial_relationships = self.calculate_spatial_relationships(objects)
        visual_analysis = {"objects": objects, "relationships": spatial_relationships}

        if self.depth_estimator:
            visual_analysis["depth"] = self.estimate_depth(image_path)
        
        return visual_analysis

    def detect_objects(self, image_path):
        return self.object_detector.detect_objects(image_path)

    def calculate_spatial_relationships(self, objects):
        relationships = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    position = "left of" if obj1["bbox"][0] < obj2["bbox"][0] else "right of"
                    vertical_position = "above" if obj1["bbox"][1] < obj2["bbox"][1] else "below"
                    relationships.append(f"{obj1['label']} is {position} and {vertical_position} {obj2['label']}")
        return relationships

    def estimate_depth(self, image_path):
        if self.depth_estimator:
            return self.depth_estimator.estimate_depth(image_path)

    def ask_spatial_question(self, image_path, question):
        if self.multimodal_model:
            return self.multimodal_model.ask_spatial_question(image_path, question)


# class VisionAgent:
#     def __init__(self, vision_api_type="openai", openai_api_key=None, anthropic_api_key=None):
#         self.vision_api_type = vision_api_type
#         self.openai_api_key = openai_api_key
#         self.anthropic_api_key = anthropic_api_key

#     def analyze_image(self, image_path):
#         """Analyze an image using the specified vision API and return its content description."""
#         if self.vision_api_type == "openai":
#             openai.api_key = self.openai_api_key
#             with open(image_path, "rb") as image_file:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4-vision",
#                     messages=[
#                         {"role": "system", "content": "Analyze the image and describe the content."}
#                     ],
#                     files={"image": image_file}
#                 )
#                 return response['choices'][0]['message']['content']

#         elif self.vision_api_type == "anthropic":
#             client = anthropic.Client(api_key=self.anthropic_api_key)
#             with open(image_path, "rb") as image_file:
#                 response = client.completions.create(
#                     model="claude-3.5-sonnet",
#                     prompt="Describe this image content.",
#                     files={"image": image_file}
#                 )
#                 return response['completion']

#     def capture_and_analyze(self, image_path):
#         """Capture and analyze an image, returning visual data for other agents."""
#         visual_data = self.analyze_image(image_path)
#         print("Vision Analysis:", visual_data)
#         return visual_data
