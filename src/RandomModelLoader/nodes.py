import os
import random
import hashlib
import json
import requests
import folder_paths
from nodes import CheckpointLoaderSimple, LoraLoader

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

# METADATA FETCHING FUNCTIONS

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def get_model_version_info(hash_value):
    """Get model info from CivitAI API using file hash"""
    try:
        api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        print(f"Error fetching model info from CivitAI: {e}")
        return {}

def parse_local_safetensors_metadata(lora_path):
    """Extract metadata directly from safetensors file"""
    if not safe_open:
        return {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            print(f"DEBUG: Safetensors metadata for {lora_path} -> {meta}")
            return meta
    except Exception as e:
        print(f"Failed reading local safetensors metadata for {lora_path}: {e}")
        return {}

def extract_trigger_words_from_metadata(meta):
    """Extract trigger words from various metadata formats"""
    triggers_found = set()

    if "trainedWords" in meta and meta["trainedWords"]:
        val = meta["trainedWords"]
        if isinstance(val, str):
            triggers_found.update(x.strip() for x in val.split(",") if x.strip())
        elif isinstance(val, list):
            triggers_found.update(val)
        elif isinstance(val, dict):
            triggers_found.update(val.keys())

    if "modelspec.trigger_phrase" in meta:
        trigger_phrase = meta["modelspec.trigger_phrase"]
        if isinstance(trigger_phrase, str) and trigger_phrase.strip():
            triggers_found.add(trigger_phrase.strip())

    return list(triggers_found)

def get_lora_metadata(lora_name):
    """Get LoRA metadata with caching, trying local first then API fallback"""
    db_path = os.path.join(os.path.dirname(__file__), 'lora_metadata_db.json')
    
    # Load existing cache
    try:
        with open(db_path, 'r') as f:
            db = json.load(f)
    except Exception:
        db = {}
    
    # Return cached result if available
    if lora_name in db:
        return db[lora_name]
    
    # Get full path to LoRA file
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not os.path.exists(lora_path):
        db[lora_name] = {"triggerWords": ""}
        try:
            with open(db_path, 'w') as f:
                json.dump(db, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata cache: {e}")
        return db[lora_name]
    
    # Try local metadata first
    meta = parse_local_safetensors_metadata(lora_path)
    local_triggers = extract_trigger_words_from_metadata(meta)
    
    # If no local triggers found, try CivitAI API
    if not local_triggers:
        try:
            LORAsha256 = calculate_sha256(lora_path)
            model_info = get_model_version_info(LORAsha256)
            if model_info.get("trainedWords"):
                local_triggers = model_info["trainedWords"]
        except Exception as e:
            print(f"Error fetching from CivitAI API: {e}")
    
    # Format trigger words as comma-separated string
    triggers_str = ", ".join(local_triggers) if local_triggers else ""
    
    # Cache the result
    db[lora_name] = {"triggerWords": triggers_str}
    try:
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")
    
    return db[lora_name]

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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "selected_checkpoint")
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

    def load_random_checkpoint(self, subfolder, seed, unique_id=None, extra_pnginfo=None):
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
        
        # Update workflow with selected checkpoint for UI display
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [subfolder, seed, selected_checkpoint]
        
        return {"ui": {"selected_checkpoint": [selected_checkpoint]}, "result": (model, clip, vae, selected_checkpoint)}
    
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "selected_lora", "trigger_words")
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

    def load_random_lora(self, model, clip, subfolder, strength_model, strength_clip, seed, unique_id=None, extra_pnginfo=None):
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
        
        # Extract trigger words from the selected LoRA
        trigger_words = ""
        try:
            metadata = get_lora_metadata(selected_lora)
            trigger_words = metadata.get("triggerWords", "")
        except Exception as e:
            print(f"Error extracting trigger words for {selected_lora}: {e}")
            trigger_words = ""
        
        # Update workflow with selected LoRA and trigger words for UI display
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [subfolder, strength_model, strength_clip, seed, selected_lora, trigger_words]
        
        return {"ui": {"selected_lora": [selected_lora], "trigger_words": [trigger_words]}, "result": (model_out, clip_out, selected_lora, trigger_words)}
    
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
