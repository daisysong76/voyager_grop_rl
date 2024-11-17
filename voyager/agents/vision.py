# TODO 3: Abstracting API Calls Using keys.js
# Instead of handling each API (OpenAI, Anthropic) within VisionAgent, use an adapter pattern to encapsulate the differences between APIs. 
# This way, each API call is defined in its own class, making it easier to add new vision APIs or swap out existing ones without modifying VisionAgent.

import openai
import anthropic
import requests
import voyager.utils as U
import requests
import os
import cv2
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

class VisionAgent:
    def __init__(
        self,
        model_name="gpt-4-turbo-vision", 
        temperature=0, 
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/vision")
        if resume:
            print(f"\033[32mLoading Vision Agent from {ckpt_dir}/vision\033[0m")
            self.vision_memory = U.load_json(f"{ckpt_dir}/vision/vision_memory.json")
        else:
            self.vision_memory = {}

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )

        # # write a great prompt for the vision agent
        # with open("/Users/daisysong/Desktop/CS194agent/Voyager_OAI/voyager/prompts/vision_template.txt", "r") as file:
        #     self.prompt = file.read()
   
        # # Initialize the ChatGPT model
        # self.llm = ChatOpenAI(
        #     model_name=model_name,
        #     temperature=temperature,
        #     request_timeout=request_timout,
        #     prompt=self.prompt
        # )

    def update_vision_memory(self, vision_data):
        self.vision_memory.append(vision_data)
        U.save_json(self.vision_memory, f"{self.ckpt_dir}/vision/vision_memory.json")
        pass
    def render_vision_message(self, image_path):
        #return HumanMessage(content=self.prompt)
        pass
    def process_vision_message(self, message):
        pass

    def analyze_image(self, image_path):
        """Send the image to GPT-4 Vision for analysis."""
        # Open the image file in binary mode
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Define the response format if not already defined
        response_format = """
        {
            "optimal_block": {
                "type": "string",
                "position": {
                    "x": "float",
                    "y": "float",
                    "z": "float"
                }
            },
            "other_blocks": [
                {
                    "type": "string",
                    "position": {
                        "x": "float",
                        "y": "float",
                        "z": "float"
                    }
                },
                ...
            ]
        }
        """

        prompt = f"""
        You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
        Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

        ### Task Instructions
        1. **Enhanced Block Detection**:
        - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
        - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
        - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
        - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

        ### Response Guidelines
        - Your responses must strictly adhere to the specified format for clear communication with the AI bot.

        ### RESPONSE FORMAT:
        {response_format}
        """

        # Prepare the messages with the prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Make the API call with the image
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages,
                files={'image': image_data},
                max_tokens=2000
            )
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

        return response

# Example usage
if __name__ == "__main__":
    vision_agent = VisionAgent()
    image_path = "/Users/daisysong/Desktop/CS194agent/Voyager_OAI/logs/visions/logs/visions/screenshot-2024-11-17T01-37-14-716Z.jpg"  # Update with path to the current image most recently captured
    insights = vision_agent.analyze_image(image_path)
    print("Vision Agent Insights:", insights)
    # save the insights to a file
    with open("vision_insights.txt", "w") as file:
        file.write(insights)




# import requests
# import voyager.utils as U
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage
# import numpy as np
# import cv2  # OpenCV for image processing

# class VisionAgent:
#     def __init__(self, model_name="gpt-4", temperature=0, request_timeout=120):
#         self.model_name = model_name
#         self.temperature = temperature
#         self.request_timeout = request_timeout
        
#         # Initialize the ChatGPT model
#         self.llm = ChatOpenAI(
#             model_name=self.model_name,
#             temperature=self.temperature,
#             request_timeout=self.request_timeout,
#         )

#     def analyze_image(self, image_path):
#         """Send the image to ChatGPT for analysis."""
#         with open(image_path, "rb") as image_file:
#             image_data = image_file.read()
        
#         # Prepare the prompt for ChatGPT
#         prompt = "Analyze the following image and provide insights: [IMAGE]"
        
#         # Send the image data to the ChatGPT API
#         response = self.llm.generate(
#             messages=[HumanMessage(content=prompt)],
#             files={"image": image_data}
#         )
        
#         return response.content

#     def estimate_depth(self, image_path):
#         """Estimate depth from the image using OpenCV or a depth estimation model."""
#         # Load the image
#         image = cv2.imread(image_path)
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Use a simple depth estimation technique (e.g., using stereo images or a pre-trained model)
#         # For demonstration, we will use a placeholder for depth estimation
#         depth_map = np.random.rand(gray.shape[0], gray.shape[1])  # Placeholder for depth map
        
#         return depth_map

#     def analyze_spatial_relationships(self, image_path):
#         """Analyze spatial relationships in the image."""
#         # Load the image
#         image = cv2.imread(image_path)
        
#         # Placeholder for spatial relationship analysis
#         # This could involve detecting objects and calculating their positions
#         # For demonstration, we will return a dummy spatial relationship
#         spatial_relationships = {
#             "object1": {"position": (100, 150), "size": (50, 50)},
#             "object2": {"position": (200, 250), "size": (60, 60)},
#             "relationship": "object1 is to the left of object2"
#         }
        
#         return spatial_relationships

#     def process_image(self, image_path):
#         """Main method to process the image and get insights from ChatGPT."""
#         insights = self.analyze_image(image_path)
#         depth = self.estimate_depth(image_path)
#         spatial_relationships = self.analyze_spatial_relationships(image_path)
        
#         # Combine insights for ChatGPT
#         combined_insights = {
#             "insights": insights,
#             "depth": depth.tolist(),  # Convert to list for JSON serialization
#             "spatial_relationships": spatial_relationships
#         }
        
#         return combined_insights

#     def render_human_message(self, image_path):
#         """Render a human message based on the image analysis."""
#         combined_insights = self.process_image(image_path)
#         observation = f"Insights from the image analysis:\n{combined_insights}\n"
#         return HumanMessage(content=observation)

# # Example usage
# if __name__ == "__main__":
#     vision_agent = VisionAgent()
#     insights = vision_agent.render_human_message("/path/to/your/image.jpg")
#     print("Vision Agent Insights:", insights.content)

# class VisionAgent:
#     def __init__(self, object_detector, depth_estimator=None, multimodal_model=None):
#         self.object_detector = object_detector
#         self.depth_estimator = depth_estimator
#         self.multimodal_model = multimodal_model

#     def analyze_image(self, image_path):
#         objects = self.detect_objects(image_path)
#         spatial_relationships = self.calculate_spatial_relationships(objects)
#         visual_analysis = {"objects": objects, "relationships": spatial_relationships}

#         if self.depth_estimator:
#             visual_analysis["depth"] = self.estimate_depth(image_path)
        
#         return visual_analysis

#     def detect_objects(self, image_path):
#         return self.object_detector.detect_objects(image_path)

#     def calculate_spatial_relationships(self, objects):
#         relationships = []
#         for i, obj1 in enumerate(objects):
#             for j, obj2 in enumerate(objects):
#                 if i != j:
#                     position = "left of" if obj1["bbox"][0] < obj2["bbox"][0] else "right of"
#                     vertical_position = "above" if obj1["bbox"][1] < obj2["bbox"][1] else "below"
#                     relationships.append(f"{obj1['label']} is {position} and {vertical_position} {obj2['label']}")
#         return relationships

#     def estimate_depth(self, image_path):
#         if self.depth_estimator:
#             return self.depth_estimator.estimate_depth(image_path)

#     def ask_spatial_question(self, image_path, question):
#         if self.multimodal_model:
#             return self.multimodal_model.ask_spatial_question(image_path, question)


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
