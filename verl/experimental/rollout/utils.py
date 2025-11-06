import numpy as np

def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids

def _internvl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive <|img_context_token|> tokens in prompt_ids for InternVL, since vLLM will expand the
    <|img_context_token|> token by num_patches * num_image_token.

    For example,
    ```
    <|img_start|><|img_context_token|><|img_context_token|>...<|img_context_token|><|img_end|>
    =>
    <|img_start|><|img_end|>
    ```
    """
    prompt_ids = np.array(prompt_ids)
    prompt_ids = prompt_ids[prompt_ids != processor.image_token_id]
    prompt_ids = prompt_ids.tolist()
    return prompt_ids

def dedup_image_tokens(prompt_ids, processor):
    if processor is not None :
        if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
            return _qwen2_5_vl_dedup_image_tokens(prompt_ids, processor)
        elif "InternVLProcessor" in processor.__class__.__name__:
            return _internvl_dedup_image_tokens(prompt_ids, processor)
        else:
            return prompt_ids
    else:
        return prompt_ids
