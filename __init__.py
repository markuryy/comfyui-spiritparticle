"""Top-level package for comfyui-spiritparticle."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """spiritparticle"""
__email__ = "comfy@markury.dev"
__version__ = "0.0.1"

from .src.FolderImageSelector.nodes import NODE_CLASS_MAPPINGS as FOLDER_IMAGE_MAPPINGS
from .src.FolderImageSelector.nodes import NODE_DISPLAY_NAME_MAPPINGS as FOLDER_IMAGE_DISPLAY_MAPPINGS
from .src.RandomModelLoader.nodes import NODE_CLASS_MAPPINGS as RANDOM_LOADER_MAPPINGS
from .src.RandomModelLoader.nodes import NODE_DISPLAY_NAME_MAPPINGS as RANDOM_LOADER_DISPLAY_MAPPINGS
from .src.ShowText.nodes import NODE_CLASS_MAPPINGS as SHOW_TEXT_MAPPINGS
from .src.ShowText.nodes import NODE_DISPLAY_NAME_MAPPINGS as SHOW_TEXT_DISPLAY_MAPPINGS
from .src.ImageBlender.nodes import NODE_CLASS_MAPPINGS as IMAGE_BLENDER_MAPPINGS
from .src.ImageBlender.nodes import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_BLENDER_DISPLAY_MAPPINGS
from .src.HighPassFilter.nodes import NODE_CLASS_MAPPINGS as HIGH_PASS_MAPPINGS
from .src.HighPassFilter.nodes import NODE_DISPLAY_NAME_MAPPINGS as HIGH_PASS_DISPLAY_MAPPINGS


# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **FOLDER_IMAGE_MAPPINGS,
    **RANDOM_LOADER_MAPPINGS,
    **SHOW_TEXT_MAPPINGS,
    **IMAGE_BLENDER_MAPPINGS,
    **HIGH_PASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FOLDER_IMAGE_DISPLAY_MAPPINGS,
    **RANDOM_LOADER_DISPLAY_MAPPINGS,
    **SHOW_TEXT_DISPLAY_MAPPINGS,
    **IMAGE_BLENDER_DISPLAY_MAPPINGS,
    **HIGH_PASS_DISPLAY_MAPPINGS
}

WEB_DIRECTORY = "./src/web"
