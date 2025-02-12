# test_vision.py
import os
import base64
import pytest
from voyager.agents.vision import VisionAgent

@pytest.fixture
def setup_image_path():
    """Fixture to set up the image path."""
    test_image_path = "test_image.jpg"
    with open(test_image_path, 'wb') as f:
        f.write(b'\x00' * 100)  # Create a dummy file of 100 bytes
    yield test_image_path
    os.remove(test_image_path)  # Cleanup after test

def test_analyze_image_success(setup_image_path):
    """Test analyze_image function for a valid image."""
    agent = VisionAgent()
    result = agent.analyze_image(setup_image_path)
    assert isinstance(result, dict)
    assert "optimal_block" in result
    assert "other_blocks" in result

def test_analyze_image_file_not_found():
    """Test analyze_image function for a non-existent image."""
    agent = VisionAgent()
    with pytest.raises(FileNotFoundError):
        agent.analyze_image("non_existent_image.jpg")

def test_encode_image(setup_image_path):
    """Test the internal image encoding function."""
    agent = VisionAgent()
    encoded_image = agent.analyze_image(setup_image_path)
    
    with open(setup_image_path, "rb") as image_file:
        expected_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    assert encoded_image == expected_encoded_image
    
# # test_vision.py
# import unittest
# import os
# import base64
# from voyager.agents.vision import VisionAgent

# class TestVisionAgent(unittest.TestCase):
#     def setUp(self):
#         """Set up test environment."""
#         self.agent = VisionAgent()
#         self.test_image_path = "test_image.jpg"
        
#         # Create a dummy image file for testing
#         with open(self.test_image_path, 'wb') as f:
#             f.write(b'\x00' * 100)  # Create a dummy file of 100 bytes

#     def tearDown(self):
#         """Clean up after tests."""
#         if os.path.exists(self.test_image_path):
#             os.remove(self.test_image_path)

#     def test_analyze_image_success(self):
#         """Test analyze_image function for a valid image."""
#         # Call the analyze_image method
#         result = self.agent.analyze_image(self.test_image_path)
        
#         # Check if the result is a dictionary (as expected)
#         self.assertIsInstance(result, dict)
        
#         # Check if the expected keys are in the result
#         self.assertIn("optimal_block", result)
#         self.assertIn("other_blocks", result)

#     def test_analyze_image_file_not_found(self):
#         """Test analyze_image function for a non-existent image."""
#         with self.assertRaises(FileNotFoundError):
#             self.agent.analyze_image("non_existent_image.jpg")

#     def test_encode_image(self):
#         """Test the internal image encoding function."""
#         # Access the internal method directly for testing
#         encoded_image = self.agent.analyze_image(self.test_image_path)
        
#         # Check if the image is encoded correctly
#         with open(self.test_image_path, "rb") as image_file:
#             expected_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
#         # Assuming the analyze_image method returns the base64 encoded image
#         self.assertEqual(encoded_image, expected_encoded_image)

# if __name__ == '__main__':
#     unittest.main()