import requests
from typing import List, Dict, Any, Optional, Union
import base64
from io import BytesIO
from PIL import Image
import numpy as np


manipulation_system_prompt = """## You are a Franka Panda robot with a parallel gripper. You can perform various tasks and output a sequence of gripper actions to accomplish a given task with images of your status. The input space, output action space and color space are defined as follows:

** Input Space **
- Each input object is represented as a 3D discrete position in the following format: [X, Y, Z]. 
- There is a red XYZ coordinate frame located in the top-left corner of the table. The X-Y plane is the table surface. 
- The allowed range of X, Y, Z is [0, 100]. 
- Objects are ordered by Y in ascending order.

** Output Action Space **
- Each output action is represented as a 7D discrete gripper action in the following format: [X, Y, Z, Roll, Pitch, Yaw, Gripper state].
- X, Y, Z are the 3D discrete position of the gripper in the environment. It follows the same coordinate system as the input object coordinates.
- The allowed range of X, Y, Z is [0, 100].
- Roll, Pitch, Yaw are the 3D discrete orientation of the gripper in the environment, represented as discrete Euler Angles. 
- The allowed range of Roll, Pitch, Yaw is [0, 120] and each unit represents 3 degrees.
- Gripper state is 0 for close and 1 for open.

** Color space **
- Each object can be described using one of the colors below:
  ["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", "magenta", "silver", "gray", "olive", "purple", "teal", "azure", "violet", "rose", "black", "white"],

** Generation Guide **
- Include the thinking process between <|think_start|> and <|think_end|>
- Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than the 7-DoF vector. Do not include any other thing, such as '"'.

"""


class ERAClient:
    """
    Client for sending requests to the ERA model server.
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:5000", timeout: int = 60):
        """
        Initialize the ERA client.
        
        Args:
            server_url: Base URL of the server (default: http://127.0.0.1:5000)
            timeout: Request timeout in seconds (default: 60)
        """
        self.server_url = server_url
        self.timeout = timeout
        self.endpoint = f"{server_url}/respond"
    
    def encode_image_to_base64(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Encode an image to base64 string.
        
        Args:
            image: Can be a file path (str), numpy array, or PIL Image
        
        Returns:
            Base64 encoded string with data URI prefix
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Encode to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Return with data URI prefix
        return f"data:image/jpeg;base64,{img_base64}"
    
    def get_response_from_url(
        self, 
        message: List[Dict[str, Any]], 
        mode: str = "self-plan",
        temperature: float = 0.01,
        max_new_tokens: int = 1024
    ) -> str:
        """
        Send a message to the server and get a response.
        
        Args:
            message: List of message dictionaries containing the conversation
            mode: Generation mode (default: "self-plan")
            temperature: Sampling temperature (default: 0.01)
            max_new_tokens: Maximum number of new tokens to generate (default: 1024)
            
        Returns:
            The model's response as a string
        """
        try:
            payload = {
                "message": message,
                "mode": mode,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            }
            response = requests.post(
                self.endpoint, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except KeyboardInterrupt:
            print("\nThe process is interrupted by user")
            if 'response' in locals():
                response.close()
            raise SystemExit(1)
        except requests.Timeout:
            print("The request timed out")
            return ""
        except Exception as e:
            print(f"The request failed: {e}")
            return ""
    
    def get_message_era(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        instruction: str,
        object_info: str = "",
        interaction_history: str = "",
        task_variation: Optional[str] = None,
        encode_to_base64: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Construct a message for the ERA model.
        
        Args:
            images: List of images (can be paths, numpy arrays, or PIL Images)
            instruction: Task instruction (e.g., "Pick up the red cube")
            object_info: Additional information about objects in the scene
            interaction_history: History of previous interactions
            task_variation: Optional task variation information
            encode_to_base64: If True, encode images to base64 (for remote servers)
            
        Returns:
            A formatted message list ready to send to the server
        """
        # Encode image if needed
        if encode_to_base64:
            image_data = self.encode_image_to_base64(images[0])
        else:
            # For local use, keep as path
            image_data = images[0] if isinstance(images[0], str) else images[0]
        
        current_message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": manipulation_system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {
                        "type": "text", 
                        "text": f"instruction: {instruction}, \n interaction_history: {interaction_history} \n additional_info: {object_info} \n Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    }
                ],
            }
        ]
        return current_message
    
    def send_manipulation_request(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        instruction: str,
        object_info: str = "",
        interaction_history: str = "",
        task_variation: Optional[str] = None,
        mode: str = "self-plan",
        temperature: float = 0.01,
        max_new_tokens: int = 1024,
        encode_to_base64: bool = True
    ) -> str:
        """
        Convenience method to construct a message and send it to the server.
        
        Args:
            images: List of images (can be paths, numpy arrays, or PIL Images)
            instruction: Task instruction
            object_info: Additional information about objects
            interaction_history: History of previous interactions
            task_variation: Optional task variation information
            mode: Generation mode
            temperature: Sampling temperature
            max_new_tokens: Maximum number of new tokens to generate
            encode_to_base64: If True, encode images to base64 (for remote servers)
            
        Returns:
            The model's response as a string
        """
        message = self.get_message_era(
            images=images,
            instruction=instruction,
            object_info=object_info,
            interaction_history=interaction_history,
            task_variation=task_variation,
            encode_to_base64=encode_to_base64
        )
        
        return self.get_response_from_url(
            message=message,
            mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ERAClient(server_url="http://127.0.0.1:22221")
    
    # Example 1: Using the convenience method
    response = client.send_manipulation_request(
        images=["/home/hanyangchen/Xarm/Xarm_test_resized.jpg"],
        instruction="Pick up the red cube",
        object_info="Object 1 is in [10, 40, 5]; Object 2 is in [30, 70, 20]; Object 3 is in [50, 50, 10]; Object 4 is in [60, 20, 30]",
        interaction_history="[]",
    )
    print("Response:", response)
    
    # # Example 2: Using the lower-level methods
    # message = client.get_message_era(
    #     images=["path/to/image.jpg"],
    #     instruction="Pick up the blue cylinder",
    #     object_info="blue cylinder at position [30, 40, 15]",
    #     interaction_history="Previous action: [25, 35, 50, 60, 40, 60, 1]"
    # )
    
    # response = client.get_response_from_url(
    #     message=message,
    #     temperature=0.0,
    #     max_new_tokens=512
    # )
    # print("Response:", response)

