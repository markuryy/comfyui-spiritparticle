import torch
import numpy as np
from PIL import Image, ImageChops
import torch.nn.functional as F

class ImageBlender:
    """
    A ComfyUI node that blends two images using Photoshop-style blend modes
    with adjustable strength/opacity.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "blend_mode": ([
                    "Normal", "Darken", "Multiply", "Color Burn", "Linear Burn",
                    "Lighten", "Screen", "Color Dodge", "Linear Dodge", 
                    "Overlay", "Soft Light", "Hard Light", "Vivid Light", 
                    "Linear Light", "Pin Light", "Add", "Color", "Difference", 
                    "Exclusion", "Hue"
                ],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "Image Processing"

    def blend_images(self, base_image, overlay_image, blend_mode, strength, mask=None):
        # Convert ComfyUI tensors to PIL Images
        base_pil = self.tensor_to_pil(base_image)
        overlay_pil = self.tensor_to_pil(overlay_image)
        
        # Resize overlay to match base image if needed
        if base_pil.size != overlay_pil.size:
            overlay_pil = overlay_pil.resize(base_pil.size, Image.LANCZOS)
        
        # Apply the selected blend mode
        blended_pil = self.apply_blend_mode(base_pil, overlay_pil, blend_mode)
        
        # Apply strength (opacity)
        if strength < 1.0:
            blended_pil = Image.blend(base_pil, blended_pil, strength)
        
        # Apply mask if provided
        if mask is not None:
            blended_pil = self.apply_mask(base_pil, blended_pil, mask)
        
        # Convert back to ComfyUI tensor format
        result_tensor = self.pil_to_tensor(blended_pil)
        
        return (result_tensor,)

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # Handle different tensor shapes
        if len(tensor.shape) == 4:
            # [batch, height, width, channels] - take first image from batch
            tensor = tensor[0]
        elif len(tensor.shape) == 2:
            # [height, width] - grayscale without channel dimension
            tensor = tensor.unsqueeze(-1)  # Add channel dimension
        elif len(tensor.shape) != 3:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 2D, 3D, or 4D tensor.")
        
        # Convert from [0,1] float to [0,255] uint8
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image based on number of channels
        if len(array.shape) == 3:
            if array.shape[2] == 3:  # RGB
                return Image.fromarray(array, 'RGB')
            elif array.shape[2] == 4:  # RGBA
                return Image.fromarray(array, 'RGBA')
            elif array.shape[2] == 1:  # Grayscale with channel dimension
                return Image.fromarray(array[:,:,0], 'L')
            else:
                raise ValueError(f"Unsupported number of channels: {array.shape[2]}. Expected 1, 3, or 4 channels.")
        else:
            # Should not reach here due to earlier checks, but handle gracefully
            return Image.fromarray(array, 'L')

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

    def mask_to_pil(self, mask_tensor):
        """Convert ComfyUI mask tensor to PIL Image"""
        # ComfyUI mask tensors are in format [batch, height, width] or [height, width]
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor[0]  # Take first mask from batch
        elif len(mask_tensor.shape) != 2:
            raise ValueError(f"Unsupported mask tensor shape: {mask_tensor.shape}. Expected 2D or 3D tensor.")
        
        # Convert from [0,1] float to [0,255] uint8
        mask_array = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image (grayscale)
        return Image.fromarray(mask_array, 'L')

    def apply_mask(self, base_image, blended_image, mask):
        """Apply mask to blend between base and blended images"""
        # Convert mask tensor to PIL Image
        mask_pil = self.mask_to_pil(mask)
        
        # Resize mask to match image dimensions if needed
        if mask_pil.size != base_image.size:
            mask_pil = mask_pil.resize(base_image.size, Image.LANCZOS)
        
        # Convert images to numpy arrays for processing
        base_array = np.array(base_image).astype(np.float32) / 255.0
        blended_array = np.array(blended_image).astype(np.float32) / 255.0
        mask_array = np.array(mask_pil).astype(np.float32) / 255.0
        
        # Expand mask to match image channels (RGB)
        if len(base_array.shape) == 3 and base_array.shape[2] == 3:
            mask_array = np.stack([mask_array, mask_array, mask_array], axis=-1)
        
        # Apply mask: result = base * (1 - mask) + blended * mask
        # Where mask = 0 (black) shows base image, mask = 1 (white) shows blended image
        result_array = base_array * (1.0 - mask_array) + blended_array * mask_array
        
        # Convert back to PIL Image
        result_array = np.clip(result_array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'RGB')

    def apply_blend_mode(self, base, overlay, mode):
        """Apply the specified blend mode to two PIL images"""
        
        # Convert to RGBA for better blending
        if base.mode != 'RGBA':
            base = base.convert('RGBA')
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        
        # Get numpy arrays for mathematical operations
        base_array = np.array(base).astype(np.float32) / 255.0
        overlay_array = np.array(overlay).astype(np.float32) / 255.0
        
        # Apply blend mode
        if mode == "Normal":
            result = overlay_array
        elif mode == "Darken":
            result = np.minimum(base_array, overlay_array)
        elif mode == "Multiply":
            result = base_array * overlay_array
        elif mode == "Color Burn":
            result = self.color_burn(base_array, overlay_array)
        elif mode == "Linear Burn":
            result = np.maximum(base_array + overlay_array - 1.0, 0.0)
        elif mode == "Lighten":
            result = np.maximum(base_array, overlay_array)
        elif mode == "Screen":
            result = 1.0 - (1.0 - base_array) * (1.0 - overlay_array)
        elif mode == "Color Dodge":
            result = self.color_dodge(base_array, overlay_array)
        elif mode == "Linear Dodge":
            result = np.minimum(base_array + overlay_array, 1.0)
        elif mode == "Overlay":
            result = self.overlay(base_array, overlay_array)
        elif mode == "Soft Light":
            result = self.soft_light(base_array, overlay_array)
        elif mode == "Hard Light":
            result = self.hard_light(base_array, overlay_array)
        elif mode == "Vivid Light":
            result = self.vivid_light(base_array, overlay_array)
        elif mode == "Linear Light":
            result = self.linear_light(base_array, overlay_array)
        elif mode == "Pin Light":
            result = self.pin_light(base_array, overlay_array)
        elif mode == "Add":
            result = np.minimum(base_array + overlay_array, 1.0)
        elif mode == "Color":
            result = self.color_blend(base_array, overlay_array)
        elif mode == "Difference":
            result = np.abs(base_array - overlay_array)
        elif mode == "Exclusion":
            result = base_array + overlay_array - 2.0 * base_array * overlay_array
        elif mode == "Hue":
            result = self.hue_blend(base_array, overlay_array)
        else:
            result = overlay_array  # Default to normal blend
        
        # Convert back to PIL Image
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result, 'RGBA').convert('RGB')

    def color_burn(self, base, overlay):
        """Color Burn blend mode"""
        result = np.zeros_like(base)
        mask = overlay > 1e-10  # Avoid division by zero
        result[mask] = 1.0 - (1.0 - base[mask]) / overlay[mask]
        result = np.clip(result, 0.0, 1.0)
        return result

    def color_dodge(self, base, overlay):
        """Color Dodge blend mode"""
        result = np.ones_like(base)
        mask = overlay < (1.0 - 1e-10)  # Avoid division by zero
        result[mask] = base[mask] / (1.0 - overlay[mask])
        result = np.clip(result, 0.0, 1.0)
        return result

    def overlay(self, base, overlay):
        """Overlay blend mode"""
        result = np.where(
            base < 0.5,
            2.0 * base * overlay,
            1.0 - 2.0 * (1.0 - base) * (1.0 - overlay)
        )
        return np.clip(result, 0.0, 1.0)

    def soft_light(self, base, overlay):
        """Soft Light blend mode"""
        # Ensure base values are not negative for sqrt operation
        base_safe = np.maximum(base, 0.0)
        result = np.where(
            overlay < 0.5,
            base - (1.0 - 2.0 * overlay) * base * (1.0 - base),
            base + (2.0 * overlay - 1.0) * (np.sqrt(base_safe) - base)
        )
        return np.clip(result, 0.0, 1.0)

    def hard_light(self, base, overlay):
        """Hard Light blend mode"""
        result = np.where(
            overlay < 0.5,
            2.0 * base * overlay,
            1.0 - 2.0 * (1.0 - base) * (1.0 - overlay)
        )
        return np.clip(result, 0.0, 1.0)

    def vivid_light(self, base, overlay):
        """Vivid Light blend mode"""
        result = np.where(
            overlay < 0.5,
            self.color_burn(base, 2.0 * overlay),
            self.color_dodge(base, 2.0 * (overlay - 0.5))
        )
        return np.clip(result, 0.0, 1.0)

    def linear_light(self, base, overlay):
        """Linear Light blend mode"""
        result = base + 2.0 * overlay - 1.0
        return np.clip(result, 0.0, 1.0)

    def pin_light(self, base, overlay):
        """Pin Light blend mode"""
        result = np.where(
            overlay < 0.5,
            np.minimum(base, 2.0 * overlay),
            np.maximum(base, 2.0 * (overlay - 0.5))
        )
        return np.clip(result, 0.0, 1.0)

    def color_blend(self, base, overlay):
        """Color blend mode - combines hue and saturation of overlay with luminance of base"""
        # Convert RGB to HSL for both images
        base_hsl = self.rgb_to_hsl(base)
        overlay_hsl = self.rgb_to_hsl(overlay)
        
        # Use hue and saturation from overlay, luminance from base
        result_hsl = np.copy(base_hsl)
        result_hsl[:,:,0] = overlay_hsl[:,:,0]  # Hue
        result_hsl[:,:,1] = overlay_hsl[:,:,1]  # Saturation
        
        # Convert back to RGB
        result = self.hsl_to_rgb(result_hsl)
        return np.clip(result, 0.0, 1.0)

    def hue_blend(self, base, overlay):
        """Hue blend mode - combines hue of overlay with saturation and luminance of base"""
        # Convert RGB to HSL for both images
        base_hsl = self.rgb_to_hsl(base)
        overlay_hsl = self.rgb_to_hsl(overlay)
        
        # Use hue from overlay, saturation and luminance from base
        result_hsl = np.copy(base_hsl)
        result_hsl[:,:,0] = overlay_hsl[:,:,0]  # Hue only
        
        # Convert back to RGB
        result = self.hsl_to_rgb(result_hsl)
        return np.clip(result, 0.0, 1.0)

    def rgb_to_hsl(self, rgb):
        """Convert RGB to HSL color space"""
        # Handle both RGB (3 channels) and RGBA (4 channels)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Luminance
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = np.zeros_like(l)
        mask = diff != 0
        s[mask] = np.where(l[mask] < 0.5, 
                          diff[mask] / (max_val[mask] + min_val[mask]),
                          diff[mask] / (2.0 - max_val[mask] - min_val[mask]))
        
        # Hue
        h = np.zeros_like(l)
        
        # Red is max
        mask_r = (max_val == r) & (diff != 0)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
        
        # Green is max
        mask_g = (max_val == g) & (diff != 0)
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2.0
        
        # Blue is max
        mask_b = (max_val == b) & (diff != 0)
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4.0
        
        h = h / 6.0  # Normalize to [0,1]
        
        # Handle alpha channel if present
        if rgb.shape[2] == 4:
            return np.stack([h, s, l, rgb[:,:,3]], axis=-1)
        else:
            # For RGB images, add a dummy alpha channel of 1.0
            alpha = np.ones_like(h)
            return np.stack([h, s, l, alpha], axis=-1)

    def hsl_to_rgb(self, hsl):
        """Convert HSL to RGB color space"""
        h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
        
        c = (1.0 - np.abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - np.abs((h * 6.0) % 2.0 - 1.0))
        m = l - c / 2.0
        
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Determine RGB based on hue sector
        h_sector = (h * 6.0).astype(int) % 6
        
        mask0 = h_sector == 0
        r[mask0] = c[mask0]
        g[mask0] = x[mask0]
        
        mask1 = h_sector == 1
        r[mask1] = x[mask1]
        g[mask1] = c[mask1]
        
        mask2 = h_sector == 2
        g[mask2] = c[mask2]
        b[mask2] = x[mask2]
        
        mask3 = h_sector == 3
        g[mask3] = x[mask3]
        b[mask3] = c[mask3]
        
        mask4 = h_sector == 4
        r[mask4] = x[mask4]
        b[mask4] = c[mask4]
        
        mask5 = h_sector == 5
        r[mask5] = c[mask5]
        b[mask5] = x[mask5]
        
        r += m
        g += m
        b += m
        
        return np.stack([r, g, b, hsl[:,:,3]], axis=-1)


# Register the node
NODE_CLASS_MAPPINGS = {
    "ImageBlender": ImageBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlender": "Image Blender (Photoshop Modes)"
}
