import os
from typing import Optional
import time
from .base import Text2SQLGeneratorBase

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")


class Text2SQLGeneratorOpenRouter(Text2SQLGeneratorBase):
    """
    OpenRouter API generator for Text2SQL benchmarking.
    Uses OpenRouter's API which provides access to multiple LLM providers.
    """
    
    # Model mapping from short names to OpenRouter model IDs
    MODEL_MAPPING = {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o", 
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "claude-3-sonnet": "anthropic/claude-3.5-sonnet",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "deepseek-v3": "deepseek/deepseek-chat",
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct"
    }
    
    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        type: str,
        experiment_folder: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        data_base_type: Optional[str] = "sqlite",
    ):
        self.data_base_type = data_base_type
        self._api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OpenRouter API key must be provided either as parameter or OPENROUTER_API_KEY environment variable")
        
        # Map short model names to full OpenRouter model IDs
        self.model_name = self.MODEL_MAPPING.get(model_name, model_name)
        self.original_model_name = model_name
        self.data_base_type = data_base_type
        
        super().__init__(
            experiment_folder=experiment_folder,
            experiment_name=experiment_name,
            type=type,
        )

    @property
    def load_client(self):
        """Initialize OpenRouter client using OpenAI SDK with custom base URL"""
        extra_headers = {}
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        return client

    @property
    def load_tokenizer(self):
        """OpenRouter doesn't provide tokenizer access"""
        pass

    @property
    def model_name_or_path(self):
        return self.model_name



    def generate(
        self,
        data_blob: dict,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        postprocess: Optional[bool] = True,
        retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs
    ) -> str:
        """
        Generate SQL from the given prompt using OpenRouter API
        
        Args:
            data_blob: Dictionary containing the prompt and other data
            temperature: Sampling temperature (0.0 for deterministic)
            max_new_tokens: Maximum number of tokens to generate
            postprocess: Whether to postprocess the output
            retries: Number of retries in case of failure
            retry_delay: Delay between retries in seconds
            **kwargs: Additional generation parameters
            
        Returns:
            Generated SQL string
        """
        prompt = data_blob["prompt"]
        max_tokens = max_new_tokens

        generation_config = {
            "temperature": temperature, 
            "max_tokens": max_tokens,
            **kwargs
        }

        if self.data_base_type == 'sqlite':
            system_prompt = (
                "You are an expert SQLite developer. Your role is to convert user questions into "
                "accurate, efficient SQL queries based on the provided database schema. Always return "
                "only the SQL query without any explanations or formatting."
            )
        elif self.data_base_type == 'postgresql':
            system_prompt = (
                "You are an expert PostgreSQL developer. Your role is to convert user questions into "
                "accurate, efficient SQL queries based on the provided database schema. Always return "
                "only the SQL query without any explanations or formatting."
            )
        elif self.data_base_type == 'wikisql':
            system_prompt = (
                "You are an expert SQLite developer. Your role is to convert user questions into "
                "accurate, efficient SQL queries based on the provided database schema. Always return "
                "only the SQL query without any explanations or formatting and use lowercase in WHERE "
                "clauses and finish the query with ; . col0, col1, col2 are the actual column names in "
                "the database, so use them in the query"
            )
        else:
            raise ValueError(f"Invalid database type: {self.data_base_type}")

        attempt = 0
        while attempt < retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **generation_config
                )

                generated_text = completion.choices[0].message.content

                return self.postprocess(output_string=generated_text) if postprocess else generated_text

            except Exception as e:
                attempt += 1
                print(f"[Attempt {attempt}] Error: {e}")
                from premsql.logger import setup_console_logger
                logger = setup_console_logger(name="[OPENROUTER_GENERATOR]")
                logger.error(f"[Attempt {attempt}] Error generating SQL with model {self.model_name}: {str(e)}")

                if attempt >= retries:
                    raise
                time.sleep(retry_delay)


    def get_available_models(self):
        """Return list of available model short names"""
        return list(self.MODEL_MAPPING.keys())
    
    def add_model_mapping(self, short_name: str, openrouter_model_id: str):
        """Add a new model mapping"""
        self.MODEL_MAPPING[short_name] = openrouter_model_id