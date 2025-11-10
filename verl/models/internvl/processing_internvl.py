# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_processing_utils import BaseImageProcessor

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"

IMG_TOKEN = "<img></img>"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "crop_to_patches": True,
            "image_wh": None,
        },

    }


def resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch

def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])

def find_closest_aspect_ratio(width, height, min_num=1, max_num=12, image_size=448):
    # llava like, from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/3f7a8da1b7a8b928b5ee229fae33cf43fd64cf31/image_processing_minicpmv.py#L257 with modification 
    original_width = width
    original_height = height

    log_ratio = math.log(original_width / original_height)
    # ratio = original_width * original_height / (image_size * image_size)
    ratio = max(original_width/image_size, 1) * max(original_height/image_size, 1)

    # Ensure multiple is not less than min_num
    multiple = max(min(math.ceil(ratio), max_num), min_num)
    
    # Generate candidate split counts, ensuring they are not less than min_num
    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if min_num <= i <= max_num:
            candidate_split_grids_nums.append(i)
    # If no suitable candidates are found, use min_num
    if not candidate_split_grids_nums:
        candidate_split_grids_nums = [min_num]
    
    candidate_grids = []
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1

    best_grid = [1, min_num]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error

    return best_grid



def pad_image_tensor(image_tensor, target_width, target_height, image_position="top_left", pad_value=2.7):
    """
    Pad the image tensor to the target width and height.
    """
    _, current_height, current_width = image_tensor.shape
    pad_width = target_width - current_width
    pad_height = target_height - current_height

    if image_position == "center":
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
    elif image_position == "top_left":
        left = 0
        right = pad_width
        top = 0
        bottom = pad_height
    else:
        raise ValueError(
            f"Invalid image_position: '{image_position}'. Must be 'center' or 'top_left'."
        )

    padded_image_tensor = F.pad(image_tensor, (left, right, top, bottom), value=pad_value)
    return padded_image_tensor

class InternVLImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values", "image_flags"]

    def __init__(
        self,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
        num_image_token: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.image_size = image_size
        self.use_thumbnail = use_thumbnail
        self.num_image_token = num_image_token

        self.image_mean = IMAGENET_MEAN
        self.image_std = IMAGENET_STD
        
    def preprocess(self, images, min_num=1, max_num=12, image_size=448, use_thumbnail=True, pad_thumbnail=False, image_position='center', **kwargs):
        """
        Preprocess the image dynamically by finding the closest aspect ratio.
        Converting the PIL image to a PyTorch tensor, resizing the image while maintaining aspect ratio and normalizing its pixel values.
        Padding the Tensor(using a special value falls outside the normalization range) to the calculated target closest aspect ratio.
        Splitting the Tensor into smaller blocks.
        Optionally add a thumbnail version of the image.

        Returns:
            list[torch.Tensor]: A list of preprocessed, normalized, and padded image patches as PyTorch Tensors.
        """
        all_patches = []
        patches_per_image = []
        for image in images:
            original_width, original_height = image.size
            target_aspect_ratio = find_closest_aspect_ratio(original_width, original_height, min_num, max_num, image_size)
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
            scale = min(target_width/ original_width, target_height / original_height)

            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((max(1, int(original_height * scale)), max(1, int(original_width * scale))), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
            image_tensor = transform(image)
            image_tensor = pad_image_tensor(image_tensor, target_width, target_height, image_position=image_position)

            pixel_values = []
            for i in range(blocks):
                start_x = (i % (target_width // image_size)) * image_size
                end_x = (i % (target_width // image_size) + 1) * image_size
                start_y = (i // (target_width // image_size)) * image_size
                end_y = ((i // (target_width // image_size)) + 1) * image_size

                split_tensor = image_tensor[:, start_y:end_y, start_x:end_x]
                assert split_tensor.shape[1] == image_size and split_tensor.shape[2] == image_size, \
                    f"split_tensor shape {split_tensor.shape} does not match expected size {(3, image_size, image_size)} for block {i}."
                pixel_values.append(split_tensor)
            assert len(pixel_values) == blocks

            if use_thumbnail and len(pixel_values) != 1:
                if pad_thumbnail:
                    transform_thumbnail = T.Compose(
                        [
                            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                            T.Resize((int(original_height * scale / target_height * image_size), int(original_width * scale / target_width * image_size)), interpolation=InterpolationMode.BICUBIC),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ]
                    )
                    thumbnail_pixel_values = transform_thumbnail(image)
                    thumbnail_pixel_values = pad_image_tensor(thumbnail_pixel_values, image_size, image_size, image_position=image_position)
                else:
                    transform_thumbnail = T.Compose(
                        [
                            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ]
                    )
                    thumbnail_pixel_values = transform_thumbnail(image)
                pixel_values.append(thumbnail_pixel_values)
            all_patches.extend(pixel_values)
            patches_per_image.append(len(pixel_values))

        if len(all_patches) == 0:
            image_inputs = {
                'pixel_values_flat': torch.zeros((1, 3, self.image_size, self.image_size)),
                'image_flags': torch.zeros((1,), dtype=torch.long),
                'image_num_patches': [],
            }
        else:
            # 2) Stack into pixel_values tensor
            pixel_values = torch.stack(all_patches)  # shape: (total_patches, 3, H, W)
            image_flags = torch.ones(pixel_values.size(0), dtype=torch.long)
            patches_per_image = torch.tensor(patches_per_image)

            image_inputs = {
                'pixel_values_flat': pixel_values,
                'image_flags': image_flags,
                'image_num_patches': patches_per_image,
            }

        return image_inputs

    def get_image_patch(self, orign_width, orign_height, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        """
        Calculate the number of image patches based on the closest aspect ratio
        and the given width and height of the original image.
        """
        if min_num < 1:
            print(f"WARNING: min_num < 0, {min_num}")
            min_num = 1
        if max_num < 1:
            print(f"WARNING: max_num < 0, {max_num}")
            max_num = 1
        if min_num > max_num:
            print(f"WARNING: min_num > max_num, ({min_num}, {max_num})")
            min_num = max_num
        if orign_width <= 1:
            print(f"WARNING: orign_width <= 0, ({orign_width})")
            orign_width = 1
        if orign_height <= 1:
            print(f"WARNING: orign_height <= 0, ({orign_height})")
            orign_height = 1
        target_aspect_ratio = find_closest_aspect_ratio(orign_width, orign_height, min_num, max_num, image_size)
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        if use_thumbnail and blocks != 1:
            blocks += 1
        return blocks


class InternVLProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternVLProcessor.__call__`] and [`~InternVLProcessor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`AutoVideoProcessor`], *optional*):
            The video processor is a required input.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch. it should be set so that:
            image_seq_length = (config.image_size // config.patch_size) ** 2 * (config.scale_factor**2)
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "AutoImageProcessor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_size: int=448,
        patch_size: int=14,
        min_dynamic_patch: int=1,
        max_dynamic_patch: int=12,
        dynamic_image_size: bool=True,
        use_thumbnail: bool=True,
        downsample_ratio: float=0.5,
        image_token: str=IMG_CONTEXT_TOKEN,
        video_token=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.patch_size = patch_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.downsample_ratio = downsample_ratio
        self.num_image_token = int((image_size // patch_size)**2 * (downsample_ratio**2))

        if image_processor is None:
            image_processor = InternVLImageProcessor(
                image_size=image_size,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                use_thumbnail=use_thumbnail,
                num_image_token=self.num_image_token
            )
        self.image_processor = image_processor

        self.image_token = image_token
        self.video_token = None # NOTE(yzx): no video support for now

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)


    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT_TOKEN]

    @property
    def video_token_id(self) -> Optional[int]:
        if self.video_token is None:
            return None
        return self.tokenizer.get_vocab().get(self.video_token, None)

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    
    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = (self.min_dynamic_patch if min_dynamic_patch
                             is None else min_dynamic_patch)
        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)

        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )
        return get_internvl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:

        num_patches = self.image_processor.get_image_patch(
            orign_width=image_width,
            orign_height=image_height,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            self.image_processor.preprocess(
                images,
                min_num=min_num,
                max_num=max_num,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        ]

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        # videos: Optional[VideoInput] = None, # NOTE(yzx) no video support for now

        image_wh = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
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

        if text is None:
            raise ValueError("You have to specify text.")
        if not isinstance(text, list):
            text = [text]

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            image_inputs = self.image_processor.preprocess(
                images,
                min_num=self.min_dynamic_patch,
                max_num=max(self.max_dynamic_patch, 1),
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
                image_wh=image_wh
            )
            patches_per_image = image_inputs['image_num_patches']
            # 3) Replace placeholders in text, if provided
            for n_patches in patches_per_image:
                token_seq = (
                    IMG_START_TOKEN +
                    IMG_CONTEXT_TOKEN * (n_patches * self.num_image_token) +
                    IMG_END_TOKEN
                )
                text = [t.replace(IMG_TOKEN, token_seq, 1) for t in text] # NOTE(yzx): copy from vllm internvl
        else:
            # No images: return a dummy tensor and zero flags
            # pixel_values = torch.zeros((1, 3, self.image_size, self.image_size))
            # image_flags = torch.zeros((1,), dtype=torch.long)
            image_inputs = {}
            updated_text = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs['text_kwargs'])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def insert_media_placeholders_from_image_wh(self, text, image_wh):
        for wh in image_wh:
            n_patches = self.image_processor.get_image_patch(wh[0], wh[1], min_num=self.min_dynamic_patch, max_num=max(self.max_dynamic_patch, 1), image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            token_seq = (
                IMG_START_TOKEN +
                IMG_CONTEXT_TOKEN * (n_patches * self.num_image_token) +
                IMG_END_TOKEN
            )
            text = text.replace(IMG_TOKEN, token_seq, 1)
        return text

    @property
    def model_input_names(self):
        # Overwritten because InternVL renames video inputs to `pixel_values` before returning
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return tokenizer_input_names + image_processor_input_names


__all__ = ["InternVLProcessor"]