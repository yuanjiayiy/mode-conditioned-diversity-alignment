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
Preprocess the Infinite Chats Taxonomy dataset to parquet format
"""

import argparse
import os
import re

import datasets
from datasets import concatenate_datasets
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
        "--local_save_dir", default="~/data/tulu3_infinite-chats-taxonomy_mix", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--ratio", default=4, help="The ratio of the Tulu3 dataset to the Infinite Chats Taxonomy dataset.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    infinite_chats_taxonomy_data_source = "liweijiang/infinite-chats-taxonomy"
    tulu3_data_source = "allenai/tulu-3-sft-mixture"
    
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, split="train")
    else:
        infinite_chats_taxonomy_dataset = datasets.load_dataset(infinite_chats_taxonomy_data_source, split="train")
        # Filter for response_type == 'multiple'
        if "response_type" in infinite_chats_taxonomy_dataset.features:
            infinite_chats_taxonomy_dataset = infinite_chats_taxonomy_dataset.filter(lambda x: x["response_type"] == "multiple").select_columns(["messages"])
        infinite_chats_taxonomy_dataset = infinite_chats_taxonomy_dataset.add_column("data_source", [infinite_chats_taxonomy_data_source] * len(infinite_chats_taxonomy_dataset))
        
        tulu3_dataset = datasets.load_dataset(tulu3_data_source, split="train").select_columns(["messages", "source"])
        tulu3_dataset = tulu3_dataset.rename_column("source", "data_source")
        
        # Filter out WildChat (included in Infinite Chats Taxonomy), OASST1, and Aya datasets (non-English)
        tulu3_dataset = tulu3_dataset.filter(lambda x: x["data_source"] not in ["ai2-adapt-dev/tulu_v3.9_wildchat_100k",
                                                                           "ai2-adapt-dev/oasst1_converted",
                                                                           "ai2-adapt-dev/tulu_v3.9_aya_100k"])
        tulu3_dataset = tulu3_dataset.shuffle(seed=42).select(range(int(len(infinite_chats_taxonomy_dataset) * args.ratio)))
        
        # concatenate the two datasets
        dataset = concatenate_datasets([infinite_chats_taxonomy_dataset, tulu3_dataset], axis=0)
    
    
    # Filter out long messages
    dataset = dataset.filter(lambda x: len(x["messages"][0]["content"]) <= 4096)

    

    # Split into 8:2 train:validation
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    # Map question to question (no extra prompt)
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["messages"][0]["content"]
            data_source = example["data_source"]
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "type": "chat",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "",  # should not be used
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "reference_score": None,  # base model avg Skyreward; populated by reference-phase script
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True).remove_columns(["messages"])
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True).remove_columns(["messages"])

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