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
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                }),
                "recursive_search": (["True", "False"],),
                "load_text_file": (["True", "False"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
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

    def select_image(self, folder_path, selection_method, seed, index, recursive_search, load_text_file, unique_id=None):
        # Always try to get image paths
        current_image_paths = self.get_image_paths(folder_path, recursive_search)
        
        # Reset conditions
        reset_needed = (
            not self.image_paths or 
            folder_path != self.last_folder or 
            recursive_search != self.last_recursive
        )
        
        if reset_needed:
            self.image_paths = current_image_paths
            self.last_folder = folder_path
            self.last_recursive = recursive_search
        
        if not self.image_paths:
            raise ValueError(f"No images found in {folder_path}")
        
        # Select image path based on the method
        if selection_method == "random":
            # Use seed for random selection
            random.seed(seed)
            selected_path = random.choice(self.image_paths)
        else:  # sequential
            # Use index parameter for sequential selection
            # This will be properly incremented by ComfyUI's control_after_generate
            image_index = index % len(self.image_paths)
            selected_path = self.image_paths[image_index]
        
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
        
        current_image_index = index % len(self.image_paths) if selection_method == "sequential" else "random"
        print(f"Selected image: {selected_path}")
        print(f"Current index: {current_image_index}")
        print(f"Text content: {text_content[:50]}..." if len(text_content) > 50 else f"Text content: {text_content}")
        
        return (img_tensor, text_content)
    
    @classmethod
    def IS_CHANGED(s, folder_path, selection_method, seed, index, recursive_search, load_text_file, unique_id=None):
        # Unique identifier to control re-execution
        if selection_method == "random":
            # For random mode, seed determines the selected image
            return f"{folder_path}_{recursive_search}_{seed}"
        else:  # sequential mode
            # For sequential mode, index determines the selected image
            return f"{folder_path}_{recursive_search}_{index}"


# Register the node
NODE_CLASS_MAPPINGS = {
    "FolderImageSelector": FolderImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderImageSelector": "Folder Image Selector"
}