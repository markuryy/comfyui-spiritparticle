# ComfyUI Spiritparticle Nodes

A node pack by spiritparticle.

Currently contains:
- A custom node for ComfyUI that selects images from a folder, along with their associated text files.

## Folder Image Selector Node

This node provides an easy way to load images from a folder along with their corresponding text captions for use in ComfyUI workflows.

### Features

- Select images from a folder with two modes:
  - **Random**: Randomly select images (controlled by seed)
  - **Sequential**: Go through images one by one in alphabetical order (using seed as an index)
- Automatically load associated text files (with the same name but .txt extension)
- Option to search recursively through subfolders

### Parameters

- **folder_path**: Path to the folder containing your images
- **selection_method**: Choose between "random" or "sequential" modes
- **seed**: 
  - In random mode: Used as a random seed to select images
  - In sequential mode: Used as an index to select images in order
  - Right-click on seed and set "control_after_generate" to:
    - "increment" for sequential access (goes to next image)
    - "randomize" for random access (selects random images)
- **recursive_search**: Whether to search in subfolders
- **load_text_file**: Whether to load the corresponding text file with the same name

### Outputs

- **image**: The selected image
- **text**: Content of the corresponding text file (if available and loaded)

## Usage

1. Add the "Folder Image Selector" node to your workflow
2. Set the folder path to your images
3. Configure the selection mode and other parameters
4. Set seed control method:
   - For sequential mode: Right-click on the seed input and set "control_after_generate" to "increment"
   - For random mode: Right-click on the seed input and set "control_after_generate" to "randomize"
5. Connect the outputs to use in your workflow

## Text Files

Text files should have the same name as the image but with a .txt extension.

Example:
- `image001.png` and `image001.txt`
- `folder/landscape.jpg` and `folder/landscape.txt`