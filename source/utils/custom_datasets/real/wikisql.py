from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download

from utils.custom_datasets.base import Text2SQLBaseDataset
from premsql.logger import setup_console_logger

logger = setup_console_logger("[WIKISQL-DATASET]")


class WikiSQLDataset(Text2SQLBaseDataset):
    def __init__(
        self,
        split: str,
        dataset_folder: Optional[Union[str, Path]] = "./data",
        data_shema_folder: str = "./data/wikisql/data_schema",
        prompt_template: str = "source/prompts/new_prompt.md",
    ):
        dataset_folder = Path(dataset_folder)
        wikisql_folder = dataset_folder / "wikisql"
        if not wikisql_folder.exists():
            raise ValueError("WikiSQL dataset not found")


        assert split in ["test"], ValueError(
            "Split should be test"
        )
        if split == "test":
            json_file_name = "test.json"
        else:
            raise ValueError("Split should be test")

        super().__init__(
            split=split,
            dataset_path=wikisql_folder,
            database_folder_name="database",
            json_file_name=json_file_name,
            data_shema_folder=data_shema_folder,
            prompt_template=prompt_template,
        )
        logger.info("Loaded WikiSQL Dataset")

        # An extra step for WikiSQL Dataset so that it can be
        # compatible with the Base dataset and Base instance

        for content in self.dataset:
            content["SQL"] = content["query"]

    def setup_dataset(
        self,
        filter_by: tuple | None = None,
        num_rows: int | None = None,
        num_fewshot: int | None = None,
        model_name_or_path: str | None = None,
        prompt_template: str | None = None,
        tokenize: bool | None = False 
    ):
        logger.info("Setting up Spider Dataset")
        return super().setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
            num_fewshot=num_fewshot
        )