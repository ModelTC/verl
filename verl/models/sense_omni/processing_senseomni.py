# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from torch import TensorType
from torchvision.transforms.functional import InterpolationMode
from transformers import BatchEncoding, WhisperFeatureExtractor
from transformers.audio_utils import AudioInput
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

AUDIO_START_TOKEN = "<audio>"
AUDIO_END_TOKEN = "</audio>"
AUDIO_CONTEXT_TOKEN = "<AUDIO_CONTEXT>"

IMG_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_T_after_cnn(L_in, dilation=1):
    for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out

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



class SenseOmniProcessor(ProcessorMixin):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    feature_extractor_class = "AutoFeatureExtractor"
    # tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: InternVLImageProcessor,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer,
        chat_template=None,
        image_size: int=448,
        patch_size: int=14,
        min_dynamic_patch: int=1,
        max_dynamic_patch: int=12,
        dynamic_image_size: bool=True,
        use_thumbnail: bool=True,
        downsample_ratio: float=0.5,
        **kwargs,
    ):

        super().__init__(
            image_processor,
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
        )

        self.image_processor = image_processor
        self.feature_extractor = feature_extractor

        self.image_size = image_size
        self.patch_size = patch_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.downsample_ratio = downsample_ratio

        self.num_image_token = int((self.image_size // self.patch_size)**2 * (self.downsample_ratio**2))

        self.image_token = IMG_CONTEXT_TOKEN
        self.audio_token = AUDIO_CONTEXT_TOKEN

    def _make_batch_input(self,
                          input_item: Optional[Union[Any, list[Any]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_patch: int,
        max_dynamic_patch: int,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:

        if len(images) == 0:
            image_inputs = {}
        else:

            image_inputs = self.image_processor.preprocess(
                images,
                min_num=min_dynamic_patch,
                max_num=max(max_dynamic_patch, 1),
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )

            pixel_values = image_inputs['pixel_values_flat']
            image_flags = image_inputs['image_flags']
            image_num_patches = image_inputs['image_num_patches']

            image_token_num = torch.tensor(
                [n * self.num_image_token for n in image_num_patches],
                dtype=torch.long)
            image_inputs = {
                "pixel_values": pixel_values,
                "image_token_num": image_token_num,
            }

            for n_patches in image_num_patches:
                token_seq = (
                    IMG_START_TOKEN +
                    IMG_CONTEXT_TOKEN * (n_patches * self.num_image_token) +
                    IMG_END_TOKEN
                )
                text = [t.replace(IMG_TOKEN, token_seq, 1) for t in text] # NOTE(yzx): copy from vllm internvl

        return text, image_inputs

    def process_audio(self, audio, return_tensors='pt', sampling_rate=16000):
        L = audio.shape[0] if audio.shape[
            0] <= 480000 else 480000  # max_length < 30s
        mel_len = L // 160
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        inputs = self.feature_extractor(audio,
                                        return_tensors=return_tensors,
                                        sampling_rate=sampling_rate)
        inputs["audio_len_after_cnn"] = torch.tensor(audio_len_after_cnn,
                                                     dtype=torch.long)
        inputs["audio_token_num"] = torch.tensor(audio_token_num,
                                                 dtype=torch.long)
        return inputs

    def _preprocess_audio(
            self,
            text: list[str],
            audios: list[AudioInput],
            sampling_rate=16000,
            return_tensors="pt") -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(audios) == 0:
            audio_inputs = {}

            audio = np.zeros((48000,), dtype=np.float32) # NOTE: too long audio will be truncated
            audio_process_values = self.processor.process_audio(audio)
            audio_values = [audio_process_values['input_features']]
            audio_len_after_cnn = [audio_process_values['audio_len_after_cnn']]
            audio_token_num = [audio_process_values['audio_token_num']]
            
            audios_values = torch.cat(audio_values, dim=0).to(dtype=torch.bfloat16)  # (1,128, 3000)
            audios_len_after_cnn = torch.stack(audio_len_after_cnn, dim=0)
            audios_token_num = torch.stack(audio_token_num, dim=0)

            audio_inputs = {
                "audio_values": audios_values,
                "audio_len_after_cnn": audios_len_after_cnn,
                "audio_token_num": audios_token_num,
            }
            
        else:
            audios_values = []
            audios_len_after_cnn = []
            audios_token_num = []

            for audio in audios:
                audio_process_values = self.process_audio(
                    audio,
                    sampling_rate=sampling_rate,
                    return_tensors=return_tensors)
                input_features = audio_process_values["input_features"]
                audio_len_after_cnn = audio_process_values[
                    "audio_len_after_cnn"]
                audio_token_num = audio_process_values["audio_token_num"]

                audios_values.append(input_features)
                audios_len_after_cnn.append(audio_len_after_cnn)
                audios_token_num.append(audio_token_num)

            audios_values = torch.cat(audios_values, dim=0).to(
                dtype=torch.bfloat16)  # (1,128, 3000)
            audios_len_after_cnn = torch.stack(audios_len_after_cnn, dim=0)
            audios_token_num = torch.stack(audios_token_num, dim=0)

            audio_inputs = {
                "audio_values": audios_values,
                "audio_len_after_cnn": audios_len_after_cnn,
                "audio_token_num": audios_token_num,
            }

            for n_patches in audios_token_num:
                token_seq = (
                    AUDIO_START_TOKEN +
                    AUDIO_CONTEXT_TOKEN * n_patches +
                    AUDIO_END_TOKEN
                )
                text = [t.replace(AUDIO_TOKEN, token_seq, 1) for t in text]

        return text, audio_inputs

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        audios: Optional[Union[AudioInput, list[AudioInput]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ):
        # Convert inputs to batch format
        text, images, audios = [
            self._make_batch_input(x) for x in (text, images, audios)
        ]

        min_dynamic_patch = min_dynamic_patch or self.min_dynamic_patch
        max_dynamic_patch = max_dynamic_patch or self.max_dynamic_patch

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
        )
        text, audio_inputs = self._preprocess_audio(
            text=text,
            audios=
            audios,  # Note: parameter name is 'audios' in _preprocess_audio
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors=return_tensors,
        )

        text_inputs = self.tokenizer(text)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
            **audio_inputs,
        }

    def get_feature_extractor(self):
        return self.feature_extractor
