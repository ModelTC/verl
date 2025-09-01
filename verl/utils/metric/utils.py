# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Metrics utils.
"""

from typing import Any

import numpy as np


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


def process_async_metrics(metrics_list) -> dict[str, Any]:
    from collections import defaultdict
    metrics = defaultdict(list)
    address_sequences = defaultdict(list)
    timing = {}

    for seq_metrics in metrics_list:
        for key, val in seq_metrics.items():
            if key == "address":
                continue
            metrics[key].append(val)

        if 'address' in seq_metrics and 'generate_sequences' in seq_metrics:
            address = seq_metrics['address']
            generate_sequences = seq_metrics['generate_sequences']
            address_sequences[address].append(generate_sequences)

    for key, val in metrics.items():
        timing[f"seq_{key}_mean"] = np.mean(val)
        timing[f"seq_{key}_max"] = np.max(val)
        timing[f"seq_{key}_min"] = np.min(val)

    # 计算每个address下generate_sequences的最大值
    address_max_values = []
    for address, sequences in address_sequences.items():
        max_val = np.max(sequences)
        address_max_values.append(max_val)
    
    # 计算所有address之间的最大值、最小值、平均值
    if address_max_values:
        timing["engine_generate_max"] = np.max(address_max_values)
        timing["engine_generate_min"] = np.min(address_max_values)
        timing["engine_generate_mean"] = np.mean(address_max_values)

    return timing
