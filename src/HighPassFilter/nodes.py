import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2

class HighPassFilter:
    """
    A ComfyUI node that applies a High Pass Filter to an image.
    High Pass Filter enhances edges and fine details by removing low-frequency information.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "desaturate": (["False", "True"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("filtered_image",)
    FUNCTION = "apply_high_pass"
    CATEGORY = "Image Processing/Filters"

    def apply_high_pass(self, image, radius, desaturate):
        # Convert ComfyUI tensor to PIL Image
        pil_image = self.tensor_to_pil(image)
        
        # Apply High Pass Filter
        filtered_pil = self.high_pass_filter(pil_image, radius)
        
        # Apply desaturation if requested
        if desaturate == "True":
            filtered_pil = self.desaturate_image(filtered_pil)
        
        # Convert back to ComfyUI tensor format
        result_tensor = self.pil_to_tensor(filtered_pil)
        
        return (result_tensor,)

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI tensors are in format [batch, height, width, channels]
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first image from batch
        
        # Convert from [0,1] float to [0,255] uint8
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if array.shape[2] == 3:  # RGB
            return Image.fromarray(array, 'RGB')
        elif array.shape[2] == 4:  # RGBA
            return Image.fromarray(array, 'RGBA')
        else:  # Grayscale
            return Image.fromarray(array[:,:,0], 'L')

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor format"""
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and normalize to [0,1]
        array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(array).unsqueeze(0)
        
        return tensor

    def high_pass_filter(self, image, radius):
        """
        Apply High Pass Filter to an image.
        
        High Pass Filter = Original Image - Low Pass Filter (Gaussian Blur)
        Then offset by 0.5 (128 in 8-bit) to center the result around middle gray.
        """
        # Convert PIL to numpy array for processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image).astype(np.float32)
        
        # Apply Gaussian blur (low pass filter)
        # Convert radius to sigma (standard deviation)
        sigma = radius / 3.0  # Typical conversion
        
        # Apply Gaussian blur to each channel
        blurred = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            blurred[:,:,i] = cv2.GaussianBlur(img_array[:,:,i], (0, 0), sigma)
        
        # High pass = Original - Blurred + 128 (offset to middle gray)
        high_pass = img_array - blurred + 128.0
        
        # Clamp values to valid range [0, 255]
        high_pass = np.clip(high_pass, 0, 255)
        
        # Convert back to PIL Image
        result_array = high_pass.astype(np.uint8)
        return Image.fromarray(result_array, 'RGB')

    def gaussian_blur_numpy(self, image, sigma):
        """
        Alternative implementation using numpy-only Gaussian blur
        (in case cv2 is not available)
        """
        from scipy.ndimage import gaussian_filter
        
        # Apply Gaussian filter to each channel
        blurred = np.zeros_like(image)
        for i in range(image.shape[2]):
            blurred[:,:,i] = gaussian_filter(image[:,:,i], sigma=sigma)
        
        return blurred

    def desaturate_image(self, image):
        """
        Convert image to grayscale using luminance-based desaturation.
        Uses the standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Apply luminance weights for proper grayscale conversion
        # These weights account for human eye sensitivity to different colors
        luminance = (img_array[:,:,0] * 0.299 +  # Red
                    img_array[:,:,1] * 0.587 +   # Green  
                    img_array[:,:,2] * 0.114)    # Blue
        
        # Create RGB image with same luminance value in all channels
        grayscale_array = np.stack([luminance, luminance, luminance], axis=2)
        
        # Convert back to uint8 and PIL Image
        grayscale_array = np.clip(grayscale_array, 0, 255).astype(np.uint8)
        return Image.fromarray(grayscale_array, 'RGB')

    def create_gaussian_kernel(self, size, sigma):
        """
        Create a Gaussian kernel for manual convolution
        (fallback if neither cv2 nor scipy is available)
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / np.sum(kernel)

    def manual_gaussian_blur(self, image, radius):
        """
        Manual implementation of Gaussian blur using convolution
        (pure numpy fallback)
        """
        sigma = radius / 3.0
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution to each channel
        from scipy import ndimage
        blurred = np.zeros_like(image)
        for i in range(image.shape[2]):
            blurred[:,:,i] = ndimage.convolve(image[:,:,i], kernel, mode='reflect')
        
        return blurred


# Register the node
NODE_CLASS_MAPPINGS = {
    "HighPassFilter": HighPassFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighPassFilter": "High Pass Filter"
}
