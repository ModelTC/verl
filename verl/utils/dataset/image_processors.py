from io import BytesIO
import math
import torch
from typing import Optional, Union, List, Tuple, Dict
from PIL import Image, ImageOps

from verl.utils.dataset.utils import dynamic_preprocess_v4, get_image_patch_v4, IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN

def adaptive_resize_image(
    image: Union[Image.Image, dict],
    max_pixels: Optional[int] = None,
    min_pixels: Optional[int] = None,
    max_aspect_ratio: Optional[float] = None,
    short_side: Optional[int] = None,
) -> Image.Image:
    """
    Adaptively resize an image based on pixel count and aspect ratio,
    and—if specified—ensure its shorter side reaches a minimum length
    by non-uniform scaling (only the short side is enlarged).

    Args:
        image: A PIL Image or a dict with a 'bytes' key containing image data.
        max_pixels: Maximum total pixels (w * h). If None, ignored.
        min_pixels: Minimum total pixels. If None, ignored.
        max_aspect_ratio: Maximum allowed aspect ratio (long/short). If None, ignored.
        short_side: Minimum length of the shorter side. If None, ignored.

    Returns:
        A resized PIL Image in RGB mode.
    """
    # 1. Load and correct orientation
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    image = ImageOps.exif_transpose(image)

    w, h = image.size

    # 2. Constrain aspect ratio
    if max_aspect_ratio is not None:
        longer, shorter = max(w, h), min(w, h)
        if longer / shorter > max_aspect_ratio:
            if w > h:
                w = int(h * max_aspect_ratio)
            else:
                h = int(w * max_aspect_ratio)

    # 3. Compute uniform scale from pixel constraints
    area = w * h
    scale = 1.0
    if max_pixels is not None and area > max_pixels:
        scale = math.sqrt(max_pixels / area)
    elif min_pixels is not None and area < min_pixels:
        scale = math.sqrt(min_pixels / area)

    # 4. Apply uniform resize for pixel/ratio constraints
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # 5. Non-uniformly enlarge the short side if needed
    if short_side is not None:
        curr_w, curr_h = image.size
        if min(curr_w, curr_h) < short_side:
            if curr_w < curr_h:
                # widen width only
                curr_w = short_side
            else:
                # height is shorter
                curr_h = short_side
            image = image.resize((curr_w, curr_h), resample=Image.LANCZOS)

    # 6. Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def preprocess_images_internvl(
    images: List[Image.Image],
    min_dynamic_patch: int = 1,
    max_dynamic_patch: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
    num_image_token: int = 256,
    text: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:
    """
    Preprocess a batch of images for InternVL:
    - Dynamically generate patches for each image.
    - Stack all patch tensors into a single pixel_values tensor.
    - Create an image_flags tensor of ones.
    - Replace each '<img></img>' placeholder in `text` with
      IMG_START_TOKEN + repeated IMG_CONTEXT_TOKEN + IMG_END_TOKEN,
      where the repetition count = num_patches_for_that_image * num_image_token.

    Args:
        images: List of PIL Image objects to process.
        min_dynamic_patch: Minimum number of patches per image.
        max_dynamic_patch: Maximum number of patches per image.
        image_size: Target size for each patch.
        use_thumbnail: If True, use thumbnail mode in dynamic preprocessing.
        num_image_token: Number of text tokens reserved per patch.
        text: Optional input string containing '<img></img>' placeholders.

    Returns:
        Tuple of:
            image_inputs: dict with keys
                - 'pixel_values': Tensor of shape (total_patches, 3, H, W)
                - 'image_flags':   Tensor of shape (total_patches,)
            updated_text: the input `text` with '<img></img>' replaced,
                          or None if `text` was None.
    """
    num_images = len(images)

    # Validate placeholder count matches actual images
    if text is not None:
        placeholder_count = text.count('<img></img>')
        if placeholder_count != num_images:
            raise ValueError(
                f"Found {placeholder_count} '<img></img>' tokens but received {num_images} images."
            )

    if num_images > 0:
        # 1) Generate and collect patches
        all_patches = []
        patches_per_image = []
        for img in images:
            patches = dynamic_preprocess_v4(
                img,
                min_num=min_dynamic_patch,
                max_num=max(max_dynamic_patch, 1),
                image_size=image_size,
                use_thumbnail=use_thumbnail
            )
            patches_per_image.append(len(patches))
            all_patches.extend(patches)

        # 2) Stack into pixel_values tensor
        pixel_values = torch.stack(all_patches)  # shape: (total_patches, 3, H, W)
        image_flags = torch.ones(pixel_values.size(0), dtype=torch.long)

        image_inputs = {
            'pixel_values': pixel_values,
            'image_flags': image_flags
        }

        # 3) Replace placeholders in text, if provided
        updated_text = text
        if updated_text is not None:
            for n_patches in patches_per_image:
                token_seq = (
                    IMG_START_TOKEN +
                    IMG_CONTEXT_TOKEN * (n_patches * num_image_token) +
                    IMG_END_TOKEN
                )
                updated_text = updated_text.replace('<img></img>', token_seq, 1)

    else:
        # No images: return a dummy tensor and zero flags
        pixel_values = torch.zeros((1, 3, image_size, image_size))
        image_flags = torch.zeros((1,), dtype=torch.long)
        image_inputs = {
            'pixel_values': pixel_values,
            'image_flags': image_flags
        }
        updated_text = text

    return image_inputs, updated_text


def get_num_image_tokens_internvl(
    image_width, 
    image_height,
    num_image_token: int=256,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True, 
):
    num_patches = get_image_patch_v4(
        image_width,
        image_height,
        min_num=min_num,
        max_num=max_num,
        image_size=image_size,
        use_thumbnail=use_thumbnail
    )

    return num_patches * num_image_token


def preprocess_images_qwenvl(
    images: List[Image.Image],
    processor,
    text: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:

    num_images = len(images)

    # Validate placeholder count matches actual images
    if text is not None:
        placeholder_count = text.count("<|vision_start|><|image_pad|><|vision_end|>")
        if placeholder_count != num_images:
            raise ValueError(
                f"Found {placeholder_count} '<image>' tokens but received {num_images} images."
            )
    updated_text = text
    if num_images > 0:
        
        image_inputs = processor.image_processor(images, return_tensors='pt')

        if text is not None:
            image_grid_thw = image_inputs['image_grid_thw']
            merge_length = processor.image_processor.merge_size**2
            index = 0
            while '<|image_pad|>' in updated_text:
                updated_text = updated_text.replace(
                    '<|image_pad|>',
                    '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            updated_text = updated_text.replace('<|placeholder|>', processor.image_token)

    else:
        # No images: return a dummy tensor and zero flags
        images = [Image.new('RGB', (28, 28), (255, 255, 255))]

        image_inputs = processor.image_processor(images, return_tensors='pt')
        
    image_inputs['image_flags'] = torch.tensor([1] * image_inputs['pixel_values'].size(0), dtype=torch.long) * int(num_images > 0)
        
    return image_inputs, updated_text