import os
import random
import glob
from PIL import Image
import numpy as np
import torch

class FolderImageSelector:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "C:/Images",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                }),
                "recursive_search": (["True", "False"],),
                "load_text_file": (["True", "False"],),
            }
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
                # Use ** for recursive search and match both lowercase and uppercase extensions
                pattern = os.path.join(folder_path, '**', f'*.{ext}')
                image_paths.extend(glob.glob(pattern, recursive=True))
                pattern = os.path.join(folder_path, '**', f'*.{ext.upper()}')
                image_paths.extend(glob.glob(pattern, recursive=True))
        else:
            for ext in extensions:
                # Direct files in the folder with both lowercase and uppercase extensions
                pattern = os.path.join(folder_path, f'*.{ext}')
                image_paths.extend(glob.glob(pattern))
                pattern = os.path.join(folder_path, f'*.{ext.upper()}')
                image_paths.extend(glob.glob(pattern))
        
        # Sort paths to ensure consistent ordering
        image_paths.sort()
        
        # Remove potential duplicates due to case-insensitive filesystems
        unique_paths = []
        normalized_paths = set()
        
        for path in image_paths:
            # Normalize the path for comparison (lowercase on Windows)
            normalized = path.lower() if os.name == 'nt' else path
            if normalized not in normalized_paths:
                normalized_paths.add(normalized)
                unique_paths.append(path)
        
        return unique_paths

    def select_image(self, folder_path, seed, recursive_search, load_text_file):
        # Get image paths
        image_paths = self.get_image_paths(folder_path, recursive_search)
        
        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")
        
        # Use seed to select image
        image_index = seed % len(image_paths)
        selected_path = image_paths[image_index]
        
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
        
        print(f"Selected image: {selected_path} (index {image_index} of {len(image_paths)})")
        print(f"Seed: {seed}")
        if text_content:
            print(f"Text content: {text_content[:50]}..." if len(text_content) > 50 else f"Text content: {text_content}")
        
        return (img_tensor, text_content)
    
    @classmethod
    def IS_CHANGED(s, folder_path, seed, recursive_search, load_text_file):
        # Changes to seed will determine the selected image
        return f"{folder_path}_{recursive_search}_{seed}"


# Register the node
NODE_CLASS_MAPPINGS = {
    "FolderImageSelector": FolderImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderImageSelector": "Folder Image Selector"
}