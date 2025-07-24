import io
import os
import sys
import random
import re
import json
import math
import copy
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

try:
    from aoss_client.client import Client as AOSSClient
    has_aoss = True
except ImportError:
    has_aoss = False


INPUT_TEMPLATE = {
    "base": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {} Please reason step by step, and put your final answer within \\boxed{{}}. Assistant: <think>\n",
    "distill": "<｜begin▁of▁sentence｜><｜User｜>{} Let's think step by step and output the final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
}


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_CONTEXT_TOKEN_QWEN = "<|image_pad|>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
CONV_START_TOKEN = "<|im_start|>"
CONV_END_TOKEN = "<|im_end|>"
ACTION_START_TOKEN = "<|action_start|>"
ACTION_END_TOKEN = "<|action_end|>"
PLUG_TOKEN = "<|plugin|>"
INTERPRETER_TOKEN = "<|interpreter|>"
PAD_TOKEN = "<|endoftext|>"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

IGNORE_INDEX = -100
IGNORE_TOKEN_ID = IGNORE_INDEX


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")

def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0:
            if len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
                # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

### gif reader ###
try:
    import imageio
except ImportError:
    print("imageio not installed")
    imageio = None

def read_frames_gif(video_path, num_frames, sample="rand", fix_start=None, client=None, min_num_frames=4):
    import cv2

    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames
### gif reader end ###

### video reader ###
try:
    from decord import VideoReader
except ImportError:
    print("decord not installed")
    VideoReader = None

def read_frames_decord(video_path, num_frames, sample="rand", fix_start=None, client=None, clip=None, min_num_frames=4):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames
### video reader end ###


def read_frames_folder(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,  # pylint: disable=W0613
    min_num_frames=4,
):
    def extract_number(file_name):
        match = re.search(r"(\d+)\.[a-zA-Z]+$", file_name)
        if match:
            return int(match.group(1))
        print("Invalid File name")
        return float("inf")  # if not found any number, return inf to ensure the file is at last

    if video_path.startswith("s3://"):
        image_list = list(client.list(video_path))
    else:
        image_list = sorted(list(os.listdir(video_path)))
    frames = sorted(image_list, key=extract_number)
    frames = [os.path.join(video_path, f) for f in frames]
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    # read frames
    frames = [Image.open(io.BytesIO(client.get(f))) for f in frames]
    return frames


class TCSLoader(object):
    """A loader class for handling different types of files (images, videos, jsonl) from local or S3 storage.

    This class provides functionality to load and process different file types using an AOSS client.
    It supports loading:
    - Images (both local and S3)
    - Videos (folder of frames, GIF, or video files) 
    - JSONL files

    Args:
        conf_path (str): Path to the AOSS configuration file
    """

    def __init__(self, conf_path):
        assert has_aoss, "please install aoss_client"
        print(f"[TCSLoader] config_path: {conf_path}")
        self.client = AOSSClient(conf_path)

    def __call__(self, fn, file_type="image", max_num_frames=-1, min_num_frames=4, sample="rand", clip=None):
        """Load and process a file based on its type.

        Args:
            fn (str): Path to the file (local or S3 path)
            file_type (str, optional): Type of file to load - "image", "video", or "jsonl". Defaults to "image".
            max_num_frames (int, optional): Maximum number of frames to load for videos. Defaults to -1.
            min_num_frames (int, optional): Minimum number of frames to load for videos. Defaults to 4.
            sample (str, optional): Frame sampling strategy for videos. Defaults to "rand".
            clip (tuple, optional): Start and end times for video clip. Defaults to None.

        Returns:
            Union[Image.Image, List[Image.Image], List[dict]]: 
                - For images: PIL Image object
                - For videos: List of PIL Image frames
                - For jsonl: List of parsed JSON objects
        """
        if file_type == "image":
            if isinstance(fn, str):
                if "s3://" in fn:
                    img_value_str = self.client.get(fn)
                    img = pil_loader(img_value_str)
                else:
                    img = Image.open(fn).convert("RGB")
            elif isinstance(fn, Image.Image):
                img = fn
            else:
                raise ValueError(f"Invalid file type: {file_type}")
            return img

        elif file_type == "video":
            if fn.endswith("/"):
                frames = read_frames_folder(
                    fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample
                )
            elif fn.endswith(".gif"):
                if imageio is None:
                    raise ImportError("imageio not installed")
                frames = read_frames_gif(
                    fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample
                )
            else:
                if VideoReader is None:
                    raise ImportError("decord not installed")
                frames = read_frames_decord(
                    fn,
                    num_frames=max_num_frames,
                    min_num_frames=min_num_frames,
                    client=self.client,
                    sample=sample,
                    clip=clip,
                )
            return frames
        
        elif file_type == "jsonl":
            file = self.client.get(fn)
            data = [json.loads(line) for line in file.splitlines()]
            return data

        else:
            raise ValueError(f"Invalid file type: {file_type}")



def find_closest_aspect_ratio_v3(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=True  # pylint: disable=W0613
):
    # llava like, from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/3f7a8da1b7a8b928b5ee229fae33cf43fd64cf31/\
    # image_processing_minicpmv.py#L257 with modification
    assert min_num == 1
    original_width, original_height = image.size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (image_size * image_size)
    multiple = min(math.ceil(ratio), max_num)
    if multiple <= 1:
        return [1, 1]
    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i > max_num:
            continue
        candidate_split_grids_nums.append(i)

    candidate_grids = []
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1
    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error

    return best_grid


def dynamic_preprocess_v3(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    target_aspect_ratio = find_closest_aspect_ratio_v3(image, min_num, max_num, image_size, use_thumbnail)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade

# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}
def build_transform(is_train, input_size, pad2square=False):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD

    if is_train:  # use data augumentation
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )

    return transform


def preprocess_internlm_v3_mm_chat(
    template_name,
    conversations,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: List[int],
    text_only: bool = False,
    use_packed_ds: bool = True,  # pylint: disable=W0613
    padding: bool = False,
    truncation: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    assert not padding
    assert template_name == "internlm2-chat-v3"

    _roles = {
        "human": "user",
        "system": "system",
        "knowledge": "knowledge",
        "gpt": "assistant",
        "plugin": "system name=<|plugin|>",
        "plugin-return": "environment name=<|plugin|>",
        "interpreter": "system name=<|interpreter|>",
        "interpreter-return": "environment name=<|interpreter|>",
    }
    conversation_start = "<|im_start|>"
    conversation_end = "<|im_end|>\n"
    num_image = len(num_image_token_list)
    if not text_only:
        new_conversations = []
        cur_image_idx = 0
        for conv in conversations:
            while "<image>" in conv["value"] and cur_image_idx < num_image:
                image_tokens = (
                    f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[cur_image_idx]}{IMG_END_TOKEN}"
                )
                conv["value"] = conv["value"].replace("<image>", image_tokens, 1)
                cur_image_idx += 1
            new_conversations.append(conv)
        conversations = new_conversations

        if cur_image_idx < num_image:
            print(
                f"WARNING: image flag mismatch: cur_image_idx {cur_image_idx} "
                f"vs. num_image {num_image}. This dataset is {ds_name}."
            )
            sys.stdout.flush()

    # Tokenize conversations
    tokens = []
    labels = []
    for _, conv in enumerate(conversations):

        role = conv['role']
        content = conv["content"]
        if role != "assistant" or (role == "assistant" and conv.get("is_input", False)):
            user_info = f"{conversation_start}{role}\n{content}{conversation_end}"
            tokenized_user = tokenizer(user_info, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            tokens.extend(tokenized_user)
            labels.extend([IGNORE_TOKEN_ID] * len(tokenized_user))
        elif role == "assistant":
            assis_start = f"{conversation_start}{role}\n"
            tokens_assistant_start = tokenizer(assis_start, return_attention_mask=False, add_special_tokens=False)[
                "input_ids"
            ]
            tokens.extend(tokens_assistant_start)
            labels.extend([IGNORE_TOKEN_ID] * len(tokens_assistant_start))
            assis_info = f"{content}{conversation_end}"
            tokenized_assistant = tokenizer(assis_info, return_attention_mask=False, add_special_tokens=False)[
                "input_ids"
            ]
            tokens.extend(tokenized_assistant)
            labels.extend(copy.deepcopy(tokenized_assistant))
        else:
            print(f"Not processed role, skip. {conv}")
    if truncation and len(tokens) > tokenizer.model_max_length:
        tokens = tokens[: tokenizer.model_max_length]
        labels = labels[: tokenizer.model_max_length]
    input_ids = torch.LongTensor([tokens])  # [1,N] to match the before
    targets = torch.LongTensor([labels])  # [1,N] to match the before
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )