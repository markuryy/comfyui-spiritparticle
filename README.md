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

## Random Model Loaders

Two nodes for randomly selecting models and LoRAs from organized subfolders.

https://github.com/user-attachments/assets/42f427e6-f111-4d77-aa25-826f9cc0c110

### Random Checkpoint Loader

Randomly selects a checkpoint from a specified subfolder within your checkpoints directory.

#### Features
- Scans checkpoint folder for subfolders (e.g., SDXL, Pony, SD1.5)
- Presents subfolder names as dropdown options
- Randomly selects a checkpoint from the chosen subfolder using seed
- Uses ComfyUI's native checkpoint loader internally
- Returns the selected checkpoint filename as a string output

#### Parameters
- **subfolder**: Dropdown list of available subfolders in your checkpoints directory
- **seed**: Seed value for reproducible random selection

#### Outputs
- **model**: The loaded model
- **clip**: The loaded CLIP model
- **vae**: The loaded VAE
- **selected_checkpoint**: String showing which checkpoint file was selected

### Random LoRA Loader

Randomly selects a LoRA from a specified subfolder within your LoRAs directory.

#### Features
- Scans LoRA folder for arbitrary subfolders (e.g., Characters, Styles, Concepts)
- Presents subfolder names as dropdown options
- Randomly selects a LoRA from the chosen subfolder using seed
- Includes standard LoRA strength controls
- Uses ComfyUI's native LoRA loader internally
- Returns the selected LoRA filename as a string output

#### Parameters
- **model**: Input model to apply LoRA to
- **clip**: Input CLIP model to apply LoRA to
- **subfolder**: Dropdown list of available subfolders in your LoRAs directory
- **strength_model**: LoRA strength for the model (default: 1.0)
- **strength_clip**: LoRA strength for CLIP (default: 1.0)
- **seed**: Seed value for reproducible random selection

#### Outputs
- **model**: The model with LoRA applied
- **clip**: The CLIP model with LoRA applied
- **selected_lora**: String showing which LoRA file was selected

### Organization Tips

To use these nodes effectively, organize your models into subfolders:

**Example Checkpoints folder structure:**
```
models/checkpoints/
├── SDXL/
│   ├── model1.safetensors
│   └── model2.safetensors
├── Pony/
│   ├── pony_model1.safetensors
│   └── pony_model2.safetensors
└── SD1.5/
    ├── realistic_model.ckpt
    └── anime_model.safetensors
```

**Example LoRAs folder structure:**
```
models/loras/
├── SDXL/
│   ├── character1.safetensors
│   └── character2.safetensors
├── Styles/
│   ├── oil_painting.safetensors
│   └── watercolor.safetensors
└── Concepts/
    ├── concept1.safetensors
    └── concept2.safetensors
```
