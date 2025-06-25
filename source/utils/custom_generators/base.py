import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import sqlparse
from tqdm.auto import tqdm
from platformdirs import user_cache_dir

from premsql.evaluator.base import BaseExecutor
from premsql.logger import setup_console_logger
from premsql.prompts import ERROR_HANDLING_PROMPT
from func_timeout import FunctionTimedOut, func_timeout

logger = setup_console_logger(name="[GENERATOR]")


class Text2SQLGeneratorBase(ABC):
    def __init__(
        self, experiment_name: str, type: str, experiment_folder: Optional[str] = None
    ):
        self.experiment_folder = (
            Path(experiment_folder)
            if experiment_folder is not None
            else Path(user_cache_dir()) / "premsql" / "experiments"
        )

        self.experiment_path = Path(self.experiment_folder) / type / experiment_name
        if not self.experiment_path.exists():
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new experiment folder: {self.experiment_path}")
        else:
            logger.info(f"Experiment folder found in: {self.experiment_path}")

        self.client = self.load_client
        self.tokenizer = self.load_tokenizer

    @property
    @abstractmethod
    def load_client(self):
        return NotImplementedError

    @property
    @abstractmethod
    def load_tokenizer(self):
        return NotImplementedError

    @property
    @abstractmethod
    def model_name_or_path(self):
        pass

    @abstractmethod
    def generate(
        self,
        data_blob: dict,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        postprocess: Optional[bool] = True,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def execution_guided_decoding(
        self,
        data_blob: dict,
        executor: BaseExecutor,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        max_retries: Optional[int] = 5,
        postprocess: Optional[bool] = True,
        **kwargs,
    ):
        error_already_found = False
        first_attempt_failed = False
        correction_successful = False
        
        for attempt in range(max_retries):
            sql = self.generate(
                data_blob=data_blob,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                postprocess=postprocess,
                **kwargs,
            )
            
            try:
                error = func_timeout(15, executor.execute_sql, args=(sql, data_blob["db_path"]))["error"] 
            except FunctionTimedOut:
                error = "Timeout"

            # Track first attempt failure
            if attempt == 0 and error:
                first_attempt_failed = True
            
            if not error:
                # If we succeeded after the first attempt failed
                if first_attempt_failed and attempt > 0:
                    correction_successful = True
                return sql, first_attempt_failed, correction_successful

            if not error_already_found:
                prompt = data_blob["prompt"].split("# SQL:")[0].strip()
                error_prompt = ERROR_HANDLING_PROMPT.format(
                    existing_prompt=prompt, sql=sql, error_msg=error
                )
                data_blob["prompt"] = error_prompt
                error_already_found = True

        return sql, first_attempt_failed, correction_successful

    def postprocess(self, output_string: str):
        sql_start_keywords = [
            r"\bSELECT\b",
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bWITH\b",
        ]

        sql_start_pattern = re.compile("|".join(sql_start_keywords), re.IGNORECASE)
        match = sql_start_pattern.search(output_string)
        if match:
            start_pos = match.start()
            sql_statement = output_string[start_pos:]
        else:
            sql_statement = output_string

        return sqlparse.format(sql_statement.split("# SQL:")[-1].strip())

    def load_results_from_folder(self):
        item_names = [item.name for item in self.experiment_path.iterdir()]

        if self.experiment_path.exists() and "predict.json" in item_names:
            return json.load(open(self.experiment_path / "predict.json", "r"))
        return None

    def generate_and_save_results(
        self,
        dataset: list[dict],
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        force: Optional[bool] = False,
        postprocess: Optional[bool] = False,
        executor: Optional[BaseExecutor] = None,
        max_retries: Optional[int] = 5,
        **kwargs,
    ) -> dict:

        existing_response = self.load_results_from_folder()
        if existing_response is not None and force == False:
            logger.info("Already results found")
            return existing_response

        to_dump = []
        count = 0
        
        # Simple correction tracking
        first_try_failed = 0
        managed_to_correct = 0
        
        for content in tqdm(dataset, total=len(dataset), desc="Generating result ..."):
            try:
                if executor is not None:
                    sql, first_attempt_failed, correction_successful = self.execution_guided_decoding(
                        data_blob=content,
                        executor=executor,
                        temperature=temperature,
                        postprocess=postprocess,
                        max_new_tokens=max_new_tokens,
                        max_retries=max_retries,
                        **kwargs,
                    )
                    
                    # Track correction stats
                    if first_attempt_failed:
                        first_try_failed += 1
                        if correction_successful:
                            managed_to_correct += 1
                            
                else:
                    sql = self.generate(
                        data_blob=content,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        postprocess=postprocess,
                        **kwargs,
                    )
                
                to_dump.append({**content, "generated": sql})
                count += 1
                
                # Save progress more frequently for large datasets
                if count % 5 == 0:
                    json.dump(to_dump, open(self.experiment_path / "predict.json", "w"), indent=4)
                    logger.info(f"Saved progress: {count}/{len(dataset)} completed")
                    
            except Exception as e:
                logger.error(f"Error processing item {count}: {str(e)}")
                # Add a fallback SQL to prevent stopping
                to_dump.append({**content, "generated": "SELECT 1;", "error": str(e)})
                count += 1
                continue

        json.dump(to_dump, open(self.experiment_path / "predict.json", "w"), indent=4)
        logger.info(f"All responses are written to: {self.experiment_path}")
        
        # Print correction statistics
        if executor is not None:
            percentage = (managed_to_correct / first_try_failed * 100) if first_try_failed > 0 else 0
            print(f"\nCorrection Statistics:")
            print(f"First try failed: {first_try_failed}")
            print(f"Managed to correct: {managed_to_correct}")
            print(f"Correction percentage: {percentage:.1f}%")
            
        return to_dump