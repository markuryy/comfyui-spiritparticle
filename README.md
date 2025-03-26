# ComfyUI Spiritparticle Nodes

A node pack by spiritparticle.

Currently contains:
- A custom node for ComfyUI that selects images from a folder, along with their associated text files.

## Folder Image Selector Node

This node provides an easy way to load images from a folder along with their corresponding text captions for use in ComfyUI workflows.

### Features

- Select images from a folder with two modes:
  - **Random**: Randomly select images (controlled by seed)
  - **Sequential**: Go through images one by one in alphabetical order
- Automatically load associated text files (with the same name but .txt extension)
- Option to search recursively through subfolders
- Remember position between workflow runs
- Reset position when needed


### Outputs

- **image**: The selected image
- **text**: Content of the corresponding text file (if available and loaded)

## Usage

1. Add the "Folder Image Selector" node to your workflow
2. Set the folder path to your images
3. Configure the selection mode and other parameters
4. Connect the outputs to use in your workflow

## Text Files

Text files should have the same name as the image but with a .txt extension.

Example:
- `image001.png` and `image001.txt`
- `folder/landscape.jpg` and `folder/landscape.txt`