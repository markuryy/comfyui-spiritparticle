# ComfyUI Spiritparticle Nodes

A node pack by spiritparticle.

Currently contains:
- A custom node for ComfyUI that selects images from a folder, along with their associated text files.

## Folder Image Selector Node

This node provides an easy way to load images from a folder along with their corresponding text captions for use in ComfyUI workflows.

https://github.com/user-attachments/assets/5a69e701-b921-46b7-88e7-b176ef48bf4a

### Features

- Selects images from a folder based on seed value
- Uses seed to deterministically select an image (seed % number_of_images)
- Automatically loads associated text files (with the same name but .txt extension)
- Option to search recursively through subfolders

### Parameters

- **folder_path**: Path to the folder containing your images
- **seed**: 
  - Used as an index to select images (seed % number_of_images)
  - Right-click on seed and set "control_after_generate" to:
    - "increment" to cycle through images sequentially
    - "randomize" to select random images each time
- **recursive_search**: Whether to search in subfolders
- **load_text_file**: Whether to load the corresponding text file with the same name

### Outputs

- **image**: The selected image
- **text**: Content of the corresponding text file (if available and loaded)

## Usage

1. Add the "Folder Image Selector" node to your workflow
2. Set the folder path to your images
3. Configure the seed control method:
   - Right-click on the seed input and set "control_after_generate" to "increment" for sequential access
   - Right-click on the seed input and set "control_after_generate" to "randomize" for random access
4. Connect the outputs to use in your workflow

## Text Files

Text files should have the same name as the image but with a .txt extension.

Example:
- `image001.png` and `image001.txt`
- `folder/landscape.jpg` and `folder/landscape.txt`
