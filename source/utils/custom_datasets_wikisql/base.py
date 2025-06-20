import json
import os
import sqlite3
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from tqdm.auto import tqdm

from premsql.logger import setup_console_logger
from premsql.prompts import BASE_TEXT2SQL_PROMPT
from premsql.utils import (
    filter_options,
    get_accepted_filters,
    get_random_few_shot_prompts,
    tokenize_fn,
)

logger = setup_console_logger(name="[DATASET]")

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    logger.warn("Ensure transformers and torch. Install using: pip install torch transformers")


IGNORE_INDEX = -100


class Text2SQLBaseInstance:
    def __init__(self, dataset: list[dict]) -> None:

        assert "question" in dataset[0], "question is required"
        assert "SQL" in dataset[0], "sql is required"
        assert "db_path" in dataset[0], "db_path is required"
        assert "db_id" in dataset[0], "db_id is required"

        self.dataset = dataset

    def __repr__(self) -> str:
        return str(json.dumps(self.dataset[:3], indent=4))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return dict(**self.dataset[idx])
    
    @staticmethod
    def to_prompt_schema(
        md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
    ) -> str:
        
        md_create = ""
        table_names = list(md.keys())
        if seed:
            np.random.seed(seed)
            np.random.shuffle(table_names)
        for table in table_names:
            md_create += f"CREATE TABLE {table} (\n"
            columns = md[table]
            if seed:
                np.random.seed(seed)
                np.random.shuffle(columns)
            for i, column in enumerate(columns):
                col_name = column["column_name"]
                # if column name has spaces, wrap it in double quotes
                if " " in col_name:
                    col_name = f'"{col_name}"'
                dtype = column["data_type"]
                col_desc = column.get("column_description", "").replace("\n", " ")
                if col_desc:
                    col_desc = f" --{col_desc}"
                if i < len(columns) - 1:
                    md_create += f"  {col_name} {dtype},{col_desc}\n"
                else:
                    # avoid the trailing comma for the last line
                    md_create += f"  {col_name} {dtype}{col_desc}\n"
            md_create += ");\n"
        return md_create

    def apply_prompt(
        self,
        prompt_template: str = "./prompts/new_prompt.md",
        data_schema_file: str = "./data/wikisql/test_wiki_sql_metadata.json",
    ):
        # print(f"Loading schema from: {data_schema_file}")
        with open(data_schema_file, "r") as f:
            md = json.load(f)

        md = md["table_metadata"]


        
        with open(prompt_template, "r") as f:
            prompt_template_content = f.read()

        for blob in tqdm(self.dataset, total=len(self.dataset), desc="Applying prompt"):
            db_id = blob["db_id"]
            single_table_md = {db_id: md[db_id]}
            schemas = self.to_prompt_schema(single_table_md)
            instructions = blob.get("instructions", "")
            final_prompt = prompt_template_content.format(
                db_type="SQLite",  # or whatever database type you're using
                user_question=blob["question"],
                question=blob["question"],  # for backward compatibility
                schemas=schemas,
                table_metadata_string=schemas,  # same as schemas
                instructions=instructions,  # add specific instructions if needed
                k_shot_prompt="",  # add few-shot examples if needed
            )
            blob["prompt"] = final_prompt
            # print(blob["prompt"])
        return self.dataset


class SupervisedDatasetForTraining(torch.utils.data.Dataset):
    @classmethod
    def load_from_pth(cls, dataset_path: Union[str, Path]):
        dataset_path = str(dataset_path)
        dataset_dict = torch.load(dataset_path)

        assert "input_ids" in dataset_dict[0], "input_ids is required"
        assert "labels" in dataset_dict[0], "labels is required"
        assert "raw" in dataset_dict[0], "raw is required"

        return cls(
            dataset=dataset_dict,
            model_name_or_path=None,
            hf_token=None,
        )

    def __init__(
        self,
        dataset: dict,
        model_name_or_path: Optional[str] = None,
        tokenize: Optional[bool] = False, 
        hf_token: Optional[str] = None,
    ):
        assert "prompt" in dataset[0], "key prompt is required"
        assert "SQL" in dataset[0], "key SQL is required"

        self.is_tokenized = False

        if model_name_or_path is not None and tokenize:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                padding_side="right",
                token=hf_token,
            )
            self.dataset = dataset

            if self.tokenizer.chat_template:
                for content in self.dataset:
                    content["prompt"] = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content["prompt"]}], tokenize=False
                    )
                logger.info("Casted dataset with model chat template")

            logger.info("Starting Tokenization ...")
            sources, targets = [], []
            for example in self.dataset:
                sources.append(example["prompt"])
                targets.append(f"{example['SQL']}{self.tokenizer.eos_token}")

            data_dict = self.preprocess(sources=sources, targets=targets)
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.is_tokenized = True

        elif "input_ids" in dataset[0] and "labels" in dataset[0]:
            self.dataset = dataset
            self.input_ids = dataset["input_ids"]
            self.labels = dataset["labels"]
            self.is_tokenized = True

        elif model_name_or_path is not None and not tokenize:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                padding_side="right",
                token=hf_token,
            )
            self.dataset = dataset
            if self.tokenizer.chat_template:
                for content in self.dataset:
                    content["prompt"] = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content["prompt"]}], tokenize=False
                    )
                logger.info("Casted dataset with model chat template")
        else:
            self.dataset = dataset

    def preprocess(self, sources: Sequence[str], targets: Sequence[str]):
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            tokenize_fn(strings, self.tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = deepcopy(input_ids)

        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if self.is_tokenized:
            return dict(
                input_ids=self.input_ids[idx],
                labels=self.labels[idx],
                raw=dict(**self.dataset[idx]),
            )
        else:
            return dict(**self.dataset[idx])

    def save_tokenized_dataset(self, path_to_save: Union[str, Path]):
        torch.save(self.dataset, str(path_to_save))
        logger.info("Dataset saved successfully in {}".format(path_to_save))


class Text2SQLBaseDataset(ABC):
    def __init__(
        self,
        split: str,
        dataset_path: Union[str, Path],
        database_folder_name: str,
        json_file_name: str,
        data_schema_file: str = "./data/wikisql/test_wiki_sql_metadata.json",
        prompt_template: str = "source/prompts/new_prompt.md",
        hf_token: Optional[str] = None,
    ):
        self.prompt_template = prompt_template
        self.dataset_path = Path(dataset_path)
        self.database_folder_name = database_folder_name
        self.dataset = json.load(open(self.dataset_path / json_file_name, "r"))
        self.data_schema_file = data_schema_file
        self.hf_token = hf_token
        assert split in ["train", "validation", "test"], ValueError(
            "Split should be either train or validation"
        )
        self.split = split

    @property
    def raw_dataset(self):
        return self._text2sql_dataset.dataset

    @property
    def filter_availables(self):
        return get_accepted_filters(data=self.dataset)

    @abstractmethod
    def setup_dataset(
        self,
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        tokenize: Optional[bool] = False,
        prompt_template: Optional[str] = None,
    ):
        for content in self.dataset:
            content["db_path"] = str(
                self.dataset_path
                / f"{self.database_folder_name}"
                / content["db_id"]
                / f"{content['db_id']}.sqlite"
            )

        if filter_by:
            self.dataset = filter_options(data=self.dataset, filter_by=filter_by)

        if num_rows:
            self.dataset = self.dataset[:num_rows]

        # Use the provided prompt_template or fall back to the instance's prompt_template
        template_to_use = prompt_template or self.prompt_template

        self.dataset = Text2SQLBaseInstance(dataset=self.dataset).apply_prompt(
            prompt_template=template_to_use,
            data_schema_file=self.data_schema_file
        )
        return SupervisedDatasetForTraining(
            dataset=self.dataset,
            model_name_or_path=model_name_or_path,
            hf_token=self.hf_token,
            tokenize=tokenize
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return dict(**self.dataset[idx])