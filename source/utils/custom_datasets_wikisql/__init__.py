from pathlib import Path
from typing import Optional, Union

from .base import Text2SQLBaseDataset
from .real.wikisql import WikiSQLDataset
# from .real.bird import BirdDataset
# from .real.spider import SpiderUnifiedDataset
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
            # "bird": BirdDataset,
            # "spider": SpiderUnifiedDataset,
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
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        tokenize: Optional[bool] = False,
        custom_db_path: Optional[str] = None
    ):
        return self._text2sql_dataset.setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            num_fewshot=num_fewshot,
            model_name_or_path=model_name_or_path,
            prompt_template=prompt_template,
            tokenize=tokenize,
            custom_db_path=custom_db_path
        )


__all__ = [
    "Text2SQLBaseDataset",
    "WikiSQLDataset", 
    "Text2SQLDataset",
]