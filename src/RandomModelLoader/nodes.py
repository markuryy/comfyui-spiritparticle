import os
import random
import folder_paths
from nodes import CheckpointLoaderSimple, LoraLoader

class RandomCheckpointLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        # Get checkpoint folder paths
        checkpoint_paths = folder_paths.get_folder_paths("checkpoints")
        
        # Find all subfolders in checkpoint directories
        subfolders = set()
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                for item in os.listdir(checkpoint_path):
                    item_path = os.path.join(checkpoint_path, item)
                    if os.path.isdir(item_path):
                        subfolders.add(item)
        
        subfolder_list = sorted(list(subfolders)) if subfolders else ["No subfolders found"]
        
        return {
            "required": {
                "subfolder": (subfolder_list,),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_random_checkpoint"
    CATEGORY = "loaders"

    def get_checkpoint_files_in_subfolder(self, subfolder_name):
        """Get all checkpoint files from the specified subfolder"""
        checkpoint_paths = folder_paths.get_folder_paths("checkpoints")
        checkpoint_files = []
        
        for checkpoint_path in checkpoint_paths:
            subfolder_path = os.path.join(checkpoint_path, subfolder_name)
            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                # Get all files in the subfolder
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    if os.path.isfile(file_path):
                        # Check if it's a valid checkpoint file
                        valid_extensions = ['.ckpt', '.safetensors', '.pt', '.pth']
                        if any(file.lower().endswith(ext) for ext in valid_extensions):
                            # Store relative path from checkpoint folder for ComfyUI
                            relative_path = os.path.join(subfolder_name, file)
                            checkpoint_files.append(relative_path)
        
        return checkpoint_files

    def load_random_checkpoint(self, subfolder, seed):
        if subfolder == "No subfolders found":
            raise ValueError("No subfolders found in checkpoints directory")
        
        # Get checkpoint files in the specified subfolder
        checkpoint_files = self.get_checkpoint_files_in_subfolder(subfolder)
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in subfolder: {subfolder}")
        
        # Use seed to select checkpoint
        random.seed(seed)
        selected_checkpoint = random.choice(checkpoint_files)
        
        print(f"Random Checkpoint Loader: Selected {selected_checkpoint} from subfolder {subfolder} (seed: {seed})")
        
        # Use the native checkpoint loader
        checkpoint_loader = CheckpointLoaderSimple()
        model, clip, vae = checkpoint_loader.load_checkpoint(selected_checkpoint)
        
        return (model, clip, vae)
    
    @classmethod
    def IS_CHANGED(s, subfolder, seed):
        return f"{subfolder}_{seed}"


class RandomLoRALoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        # Get LoRA folder paths
        lora_paths = folder_paths.get_folder_paths("loras")
        
        # Find all subfolders in LoRA directories
        subfolders = set()
        for lora_path in lora_paths:
            if os.path.exists(lora_path):
                for item in os.listdir(lora_path):
                    item_path = os.path.join(lora_path, item)
                    if os.path.isdir(item_path):
                        subfolders.add(item)
        
        subfolder_list = sorted(list(subfolders)) if subfolders else ["No subfolders found"]
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subfolder": (subfolder_list,),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_random_lora"
    CATEGORY = "loaders"

    def get_lora_files_in_subfolder(self, subfolder_name):
        """Get all LoRA files from the specified subfolder"""
        lora_paths = folder_paths.get_folder_paths("loras")
        lora_files = []
        
        for lora_path in lora_paths:
            subfolder_path = os.path.join(lora_path, subfolder_name)
            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                # Get all files in the subfolder
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    if os.path.isfile(file_path):
                        # Check if it's a valid LoRA file
                        valid_extensions = ['.safetensors', '.ckpt', '.pt', '.pth']
                        if any(file.lower().endswith(ext) for ext in valid_extensions):
                            # Store relative path from lora folder for ComfyUI
                            relative_path = os.path.join(subfolder_name, file)
                            lora_files.append(relative_path)
        
        return lora_files

    def load_random_lora(self, model, clip, subfolder, strength_model, strength_clip, seed):
        if subfolder == "No subfolders found":
            raise ValueError("No subfolders found in loras directory")
        
        # Get LoRA files in the specified subfolder
        lora_files = self.get_lora_files_in_subfolder(subfolder)
        
        if not lora_files:
            raise ValueError(f"No LoRA files found in subfolder: {subfolder}")
        
        # Use seed to select LoRA
        random.seed(seed)
        selected_lora = random.choice(lora_files)
        
        print(f"Random LoRA Loader: Selected {selected_lora} from subfolder {subfolder} (seed: {seed})")
        
        # Use the native LoRA loader
        lora_loader = LoraLoader()
        model_out, clip_out = lora_loader.load_lora(model, clip, selected_lora, strength_model, strength_clip)
        
        return (model_out, clip_out)
    
    @classmethod
    def IS_CHANGED(s, model, clip, subfolder, strength_model, strength_clip, seed):
        return f"{subfolder}_{strength_model}_{strength_clip}_{seed}"


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "RandomCheckpointLoader": RandomCheckpointLoader,
    "RandomLoRALoader": RandomLoRALoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomCheckpointLoader": "Random Checkpoint Loader",
    "RandomLoRALoader": "Random LoRA Loader"
}
