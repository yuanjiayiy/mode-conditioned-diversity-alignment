# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/dr-tulu-data", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "rl-research/dr-tulu-sft-data"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, split="train")
    else:
        dataset = datasets.load_dataset(data_source, split="train")

    # Filter for source == 'openscholar'
    if "source" in dataset.features:
        filtered_dataset = dataset.filter(lambda x: x["source"] == "openscholar")
    else:
        filtered_dataset = dataset

    # Split into 8:2 train:validation
    split_dataset = filtered_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # Map question to question (no extra prompt)
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            type = example["type"]
            response = example["conversations"][-1]["content"]
            # conversations = example["conversations"]
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "type": type,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": response,  # should not be used
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    print(f"Saving preprocessed dataset to {local_save_dir}")

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)