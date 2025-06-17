from pathlib import Path
from typing import Optional, Union

from .base import Text2SQLBaseDataset
from .real.bird import BirdDataset
from .real.spider import SpiderUnifiedDataset
from .real.wikisql import WikiSQLDataset
from premsql.utils import get_accepted_filters


class Text2SQLDataset:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        dataset_folder: Optional[Union[str, Path]] = "./data",
        prompt_template: str = "source/prompts/new_prompt.md",
        **kwargs
    ):
        assert dataset_name in ["bird", "spider", "wikisql"], ValueError(
            "Dataset should be one of bird, spider, wikisql"
        )
        dataset_mapping = {
            "bird": BirdDataset,
            "spider": SpiderUnifiedDataset,
            "wikisql": WikiSQLDataset,
        }
        self._text2sql_dataset: Text2SQLBaseDataset = dataset_mapping[dataset_name](
            split=split,
            dataset_folder=dataset_folder,
            prompt_template=prompt_template,
            **kwargs
        )

    @property
    def raw_dataset(self):
        return self._text2sql_dataset.dataset

    @property
    def filter_availables(self):
        return get_accepted_filters(data=self._text2sql_dataset.dataset)

    def setup_dataset(
        self,
        filter_by: tuple | None = None,
        num_rows: int | None = None,
        num_fewshot: int | None = None,
        model_name_or_path: str | None = None,
        prompt_template: str | None = None,
        tokenize: bool | None = False 
    ):
        return self._text2sql_dataset.setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
            num_fewshot=num_fewshot
        )


__all__ = [
    "StandardDataset",
    "GretelAIDataset",
    "SpiderUnifiedDataset",
    "BirdDataset",
    "DomainsDataset",
    "Text2SQLDataset",
]