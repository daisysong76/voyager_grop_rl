import os
import json
import base64
from datetime import datetime
from pathlib import Path
import logging
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class VisionAgent:
    def __init__(
        self,
        model_name="gpt-4-mini",
        temperature=0,
        request_timeout=120,  # Corrected parameter name
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        """
        Initialize the VisionAgent.

        Args:
            model_name (str): Name of the language model to use.
            temperature (float): Sampling temperature.
            request_timeout (int): Timeout for API requests in seconds.
            ckpt_dir (str): Directory to store checkpoints and vision data.
            resume (bool): Whether to resume from existing vision_memory.json.
            chat_log (bool): Enable or disable chat logging.
            execution_error (bool): Whether to raise exceptions on execution errors.
        """
        logging.info("Initializing VisionAgent.")
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.chat_log = chat_log
        self.execution_error = execution_error

        # Ensure the vision directory exists
        vision_dir = self.ckpt_dir / "vision"
        try:
            vision_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {vision_dir}")
        except Exception as e:
            logging.error(f"Error creating directory {vision_dir}: {e}")
            raise

        memory_file_path = vision_dir / "vision_memory.json"

        if resume:
            logging.info(f"Loading Vision Agent from {vision_dir}")
            if memory_file_path.exists():
                self.vision_memory = self._load_json(memory_file_path)
                logging.info("Loaded vision_memory.json successfully.")
            else:
                logging.warning("vision_memory.json not found. Initializing empty memory.")
                self.vision_memory = []
        else:
            self.vision_memory = []
            logging.info("Initialized empty vision_memory.")

        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )
        logging.info("ChatOpenAI model initialized.")

    def _load_json(self, file_path):
        """
        Load JSON data from a file.

        Args:
            file_path (Path): Path to the JSON file.

        Returns:
            list: Loaded JSON data as a list. Returns an empty list on failure.
        """
        try:
            with file_path.open("r") as f:
                data = json.load(f)
            logging.info(f"Successfully loaded JSON from {file_path}")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}")
            return []
        except Exception as e:
            logging.error(f"Error loading JSON from {file_path}: {e}")
            return []

    def _save_json(self, data, file_path):
        """
        Save JSON data to a file.

        Args:
            data (list): Data to save.
            file_path (Path): Path to the JSON file.
        """
        try:
            with file_path.open("w") as f:
                json.dump(data, f, indent=4)
            logging.info(f"Successfully saved JSON to {file_path}")
        except Exception as e:
            logging.error(f"Error saving JSON to {file_path}: {e}")
            if self.execution_error:
                raise

    def update_vision_memory(self, vision_data):
        """
        Update vision memory with new data and save to file.

        Args:
            vision_data (dict): New vision data to add.
        """
        logging.info("Updating vision memory.")

        # Add a timestamp to the new data
        vision_data["timestamp"] = datetime.utcnow().isoformat()

        # Path to vision memory file
        memory_file_path = self.ckpt_dir / "vision" / "vision_memory.json"

        # Load existing memory if the file exists
        if memory_file_path.exists():
            self.vision_memory = self._load_json(memory_file_path)
        else:
            self.vision_memory = []

        # Append the new data to the vision memory
        self.vision_memory.append(vision_data)

        # Save the updated memory back to the JSON file
        self._save_json(self.vision_memory, memory_file_path)

        logging.info("New analysis added to vision_memory.json.")

    def render_vision_message(self, image_path):
        """
        Render a vision message based on the image path.

        Args:
            image_path (str): Path to the image.
        """
        # TODO: Implement this method based on project requirements
        pass

    def process_vision_message(self, message):
        """
        Process a vision message.

        Args:
            message (str): The message to process.
        """
        # TODO: Implement this method based on project requirements
        pass

    def get_insights(self, voxels, block_records, entities):
        """
        Get insights based on voxels, block records, and entities.

        Args:
            voxels (list): Voxel data.
            block_records (list): Block records.
            entities (list): Entities data.
        """
        # TODO: Implement this method based on project requirements
        pass

    def get_vision_memory(self):
        """
        Get the current vision memory.

        Returns:
            list: Current vision memory.
        """
        return self.vision_memory

    def analyze_image(self, image_path):
        """
        Send the image to GPT-4 Vision or similar models for analysis.

        Args:
            image_path (str): Path to the image to analyze.

        Returns:
            dict or None: The analysis result or None if an error occurred.
        """
        logging.info(f"Analyzing image: {image_path}")

        image_path = Path(image_path).resolve()

        # Validate image path
        if not image_path.is_file():
            error_message = f"The image path {image_path} does not exist or is not a file."
            logging.error(error_message)
            if self.execution_error:
                raise ValueError(error_message)
            return None

        # Function to encode the image
        def encode_image(image_path):
            try:
                with image_path.open("rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logging.info(f"Image {image_path} encoded successfully.")
                return encoded
            except Exception as e:
                logging.error(f"Error encoding image {image_path}: {e}")
                raise

        # Encode the image to a base64 string
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            logging.error(f"Failed to encode image: {e}")
            if self.execution_error:
                raise RuntimeError(f"Failed to encode image: {e}")
            return None

        # Define the response format
        response_format = """
        {
            "optimal_block": {
                "type": "string",
                "position": {
                    "x": "float",
                    "y": "float",
                    "z": "float"
                },
                "accessibility": "boolean"
            },
            "other_blocks": [
                {
                    "type": "string",
                    "position": {
                        "x": "float",
                        "y": "float",
                        "z": "float"
                    },
                    "accessibility": "boolean"
                }
            ],
            "spatial_reasoning": {
                "relative_positions": [
                    {
                        "block": "string",
                        "relative_position": {
                            "x": "float",
                            "y": "float",
                            "z": "float"
                        }
                    }
                ]
            }
        }
        """

        # Define the prompt
        prompt = f"""
        You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
        Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

        ### Task Instructions
        1. **Enhanced Block Detection**:
        - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
        - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
        - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
        - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

        2. **JSON Requirements**:
        - Return your output **strictly as a valid JSON object**.
        - Do not include any explanations, comments, or text outside the JSON structure.
        - The JSON must follow the format exactly as specified below.

        ### RESPONSE FORMAT:
        {response_format}

        Only respond with the JSON object. Do not include anything else in your output.
        """

        # Prepare messages using LangChain's BaseMessage classes
        messages = [
            HumanMessage(content=prompt),
            HumanMessage(content=f"Analyze the following image (base64 encoded): data:image/jpeg;base64,{base64_image}")
        ]

        try:
            # Invoke the model
            response = self.llm(messages)  # Adjusted based on LangChain's API
            response_content = response.content  # Assuming response has a 'content' attribute

            # Parse the JSON response
            response_data = json.loads(response_content)
            logging.info(f"VisionAgent.analyze_image response: {response_data}")

            # Add image path to the response data
            response_data["image_path"] = str(image_path)

            # Update vision memory with the new data
            self.update_vision_memory(response_data)
            logging.info("VisionAgent.analyze_image updated vision_memory.json.")

            return response_data
        except json.JSONDecodeError as e:
            error_message = f"JSON decoding failed: {e}"
            logging.error(error_message)
            if self.execution_error:
                raise RuntimeError(error_message)
            return None
        except Exception as e:
            error_message = f"API call failed: {e}"
            logging.error(error_message)
            if self.execution_error:
                raise RuntimeError(error_message)
            return None
