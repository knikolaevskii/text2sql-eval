from pathlib import Path
from typing import Optional, Union

from premsql.logger import setup_console_logger
from utils.custom_datasets_wikisql.base import Text2SQLBaseDataset

logger = setup_console_logger("[WIKISQL-DATASET]")


class WikiSQLDataset(Text2SQLBaseDataset):
    def __init__(
        self,
        split: str,
        dataset_folder: Optional[Union[str, Path]] = "./data",
        data_schema_file: str = None,
        hf_token: Optional[str] = None,
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

        # Set default data_schema_file if not provided
        if data_schema_file is None:
            data_schema_file = str(wikisql_folder / "test_wiki_sql_metadata.json")

        super().__init__(
            split=split,
            dataset_path=wikisql_folder,
            database_folder_name="database",
            json_file_name=json_file_name,
            data_schema_file=data_schema_file,
            hf_token=hf_token,
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
        tokenize: bool | None = False,
        custom_db_path: str | None = None 
    ):
        logger.info("Setting up WikiSQL Dataset")
    
        # Call parent setup
        result = super().setup_dataset(
            filter_by=filter_by,
            num_rows=num_rows,
            num_fewshot=num_fewshot,
            model_name_or_path=model_name_or_path,
            tokenize=tokenize,
            prompt_template=prompt_template,
        )
        
        # Override database paths if custom path provided
        if custom_db_path:
            for content in self.dataset:
                content["db_path"] = custom_db_path
        
        return result