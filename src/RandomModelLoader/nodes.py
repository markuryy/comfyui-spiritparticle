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

def get_trigger_from_txt_file(lora_path, seed=0):
    """Get trigger words from txt file with same name as LoRA file"""
    try:
        # Get the base name without extension and construct txt file path
        base_name = os.path.splitext(lora_path)[0]
        txt_path = base_name + ".txt"
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                
            if file_content:
                # Check if file contains multiple trigger word sets separated by ---
                if "---" in file_content:
                    # Split by delimiter and select one randomly using the same seed
                    sections = file_content.split("---")
                    # Strip whitespace from each section (but keep empty ones as empty strings)
                    sections = [section.strip() for section in sections]
                    # Use seed to select section (same logic as LoRA selection)
                    section_index = seed % len(sections)
                    selected_content = sections[section_index]
                    print(f"DEBUG: Multiple trigger word sets found in {txt_path}: {len(sections)} sections, selected index {section_index}")
                    print(f"DEBUG: Selected trigger words: {selected_content}")
                    return selected_content
                else:
                    # No delimiter found, use entire content (backward compatibility)
                    print(f"DEBUG: Found trigger words in txt file {txt_path}: {file_content}")
                    return file_content
        return ""
    except Exception as e:
        print(f"Error reading txt file for {lora_path}: {e}")
        return ""

def get_lora_metadata(lora_name, seed=0):
    """Get LoRA metadata with caching, priority: txt file > CivitAI > safetensors metadata > none"""
    db_path = os.path.join(os.path.dirname(__file__), 'lora_metadata_db.json')
    
    # Load existing cache
    try:
        with open(db_path, 'r') as f:
            db = json.load(f)
    except Exception:
        db = {}
    
    # Create cache key that includes seed for txt file based triggers
    cache_key = f"{lora_name}_seed_{seed}"
    
    # Get full path to LoRA file
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not os.path.exists(lora_path):
        result = {"triggerWords": ""}
        db[cache_key] = result
        try:
            with open(db_path, 'w') as f:
                json.dump(db, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata cache: {e}")
        return result
    
    # Check if we have txt file first to determine caching strategy
    base_name = os.path.splitext(lora_path)[0]
    txt_path = base_name + ".txt"
    has_txt_file = os.path.exists(txt_path)
    
    # For txt files with delimiters, we need seed-specific caching
    # For other sources, we can use general caching
    if has_txt_file:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
            has_delimiters = "---" in file_content
        except:
            has_delimiters = False
        
        if has_delimiters:
            # Use seed-specific cache key for delimited txt files
            if cache_key in db:
                return db[cache_key]
        else:
            # Use general cache key for non-delimited txt files
            general_key = f"{lora_name}_general"
            if general_key in db:
                return db[general_key]
    else:
        # Use general cache key for non-txt sources
        general_key = f"{lora_name}_general"
        if general_key in db:
            return db[general_key]
    
    triggers_str = ""
    
    # Priority 1: Try txt file first (highest priority)
    txt_triggers = get_trigger_from_txt_file(lora_path, seed)
    if txt_triggers:
        triggers_str = txt_triggers
    else:
        # Priority 2: Try CivitAI API
        try:
            LORAsha256 = calculate_sha256(lora_path)
            model_info = get_model_version_info(LORAsha256)
            if model_info.get("trainedWords"):
                api_triggers = model_info["trainedWords"]
                if isinstance(api_triggers, list):
                    triggers_str = ", ".join(api_triggers)
                elif isinstance(api_triggers, str):
                    triggers_str = api_triggers
        except Exception as e:
            print(f"Error fetching from CivitAI API: {e}")
        
        # Priority 3: If still no triggers, try local safetensors metadata
        if not triggers_str:
            meta = parse_local_safetensors_metadata(lora_path)
            local_triggers = extract_trigger_words_from_metadata(meta)
            triggers_str = ", ".join(local_triggers) if local_triggers else ""
    
    # Cache the result with appropriate key
    result = {"triggerWords": triggers_str}
    if has_txt_file and has_delimiters:
        db[cache_key] = result  # Seed-specific cache
    else:
        general_key = f"{lora_name}_general"
        db[general_key] = result  # General cache
    
    try:
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")
    
    return result

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
        
        return (model, clip, vae, selected_checkpoint)
    
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
        
        # Extract trigger words from the selected LoRA using the same seed
        trigger_words = ""
        try:
            metadata = get_lora_metadata(selected_lora, seed)
            trigger_words = metadata.get("triggerWords", "")
        except Exception as e:
            print(f"Error extracting trigger words for {selected_lora}: {e}")
            trigger_words = ""
        
        return (model_out, clip_out, selected_lora, trigger_words)
    
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
