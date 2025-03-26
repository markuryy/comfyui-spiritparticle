import os
import random
import glob
from PIL import Image
import numpy as np
import torch

class FolderImageSelector:
    _instance = None

    def __new__(cls):
        # Implement a singleton pattern to ensure consistent state across executions
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize if not already initialized
        if not hasattr(self, 'initialized'):
            self.current_index = 0
            self.image_paths = []
            self.last_folder = ""
            self.last_recursive = False
            self.initialized = True
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "C:/Images",
                }),
                "selection_method": (["random", "sequential"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                }),
                "recursive_search": (["True", "False"],),
                "remember_position": (["True", "False"],),
                "load_text_file": (["True", "False"],),
                "reset_position": (["False", "True"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "select_image"
    CATEGORY = "Image"

    def get_image_paths(self, folder_path, recursive):
        """Get all image paths from the specified folder"""
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        # Define image extensions to look for
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        
        image_paths = []
        
        # Determine pattern based on recursive flag
        if recursive == "True":
            for ext in extensions:
                # Use ** for recursive search
                pattern = os.path.join(folder_path, '**', f'*.{ext}')
                image_paths.extend(glob.glob(pattern, recursive=True))
                # Also check for uppercase extensions
                pattern = os.path.join(folder_path, '**', f'*.{ext.upper()}')
                image_paths.extend(glob.glob(pattern, recursive=True))
        else:
            for ext in extensions:
                # Direct files in the folder
                pattern = os.path.join(folder_path, f'*.{ext}')
                image_paths.extend(glob.glob(pattern))
                # Also check for uppercase extensions
                pattern = os.path.join(folder_path, f'*.{ext.upper()}')
                image_paths.extend(glob.glob(pattern))
        
        # Sort paths to ensure consistent ordering for sequential mode
        image_paths.sort()
        
        return image_paths

    def select_image(self, folder_path, selection_method, seed, recursive_search, remember_position, load_text_file, reset_position):
        # Always try to get image paths
        current_image_paths = self.get_image_paths(folder_path, recursive_search)
        
        # Reset conditions
        reset_needed = (
            not self.image_paths or 
            folder_path != self.last_folder or 
            recursive_search != self.last_recursive or 
            reset_position == "True"
        )
        
        if reset_needed:
            self.image_paths = current_image_paths
            self.last_folder = folder_path
            self.last_recursive = recursive_search
            
            # Reset index when folder changes or reset is requested
            self.current_index = 0
        
        if not self.image_paths:
            raise ValueError(f"No images found in {folder_path}")
        
        # Select image path based on the method
        if selection_method == "random":
            random.seed(seed)
            selected_path = random.choice(self.image_paths)
            # In random mode, don't affect sequential indices
        else:  # sequential
            if remember_position == "False" or reset_position == "True":
                # Always start from the beginning
                selected_path = self.image_paths[0]
                self.current_index = 1  # Set to 1 to prepare for next image
            else:
                # Ensure index is within bounds
                self.current_index = self.current_index % len(self.image_paths)
                selected_path = self.image_paths[self.current_index]
                
                # Increment index for next time
                self.current_index = (self.current_index + 1) % len(self.image_paths)
        
        # Load and convert the image to ComfyUI format
        img = Image.open(selected_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        
        # Handle text file loading
        text_content = ""
        if load_text_file == "True":
            txt_path = os.path.splitext(selected_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                except Exception as e:
                    print(f"Error reading text file {txt_path}: {e}")
                    text_content = ""
        
        print(f"Selected image: {selected_path}")
        print(f"Loaded text: {text_content}")
        print(f"Current index: {self.current_index}")
        
        return (img_tensor, text_content)
    
    @classmethod
    def IS_CHANGED(s, folder_path, selection_method, seed, recursive_search, remember_position, load_text_file, reset_position):
        # Get singleton instance to access current_index
        instance = FolderImageSelector()
        
        # Unique identifier to control re-execution
        if selection_method == "random":
            return seed  # Unique for each seed in random mode
        else:  # sequential mode
            if remember_position == "True" and reset_position == "False":
                # Include current_index to force re-execution with each workflow run
                return f"{folder_path}_{recursive_search}_{remember_position}_{reset_position}_{instance.current_index}"
            else:
                # If not remembering position, just use the other parameters
                return f"{folder_path}_{recursive_search}_{remember_position}_{reset_position}"


# Register the node
NODE_CLASS_MAPPINGS = {
    "FolderImageSelector": FolderImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderImageSelector": "Folder Image Selector"
}