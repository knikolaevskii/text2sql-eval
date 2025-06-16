import os
from typing import Optional

from premsql.generators.base import Text2SQLGeneratorBase

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")


class Text2SQLGeneratorAPI(Text2SQLGeneratorBase):
    """
    Generic API generator for Text2SQL benchmarking.
    Works with any local API endpoint that accepts OpenAI-compatible requests.
    No API keys required - designed for local APIs.
    """
    
    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        type: str,
        api_base_url: str,
        experiment_folder: Optional[str] = None,
    ):
        """
        Initialize the API generator
        
        Args:
            model_name: Name/identifier of the model to use
            experiment_name: Name of the experiment
            type: Type of the experiment
            api_base_url: The base URL of the local API (e.g., "http://localhost:8000/v1")
            experiment_folder: Optional folder for experiments
        """
        self.model_name = model_name
        self.api_base_url = api_base_url
        
        super().__init__(
            experiment_folder=experiment_folder,
            experiment_name=experiment_name,
            type=type,
        )

    @property
    def load_client(self):
        """Initialize OpenAI client with custom base URL and dummy API key"""
        client = OpenAI(
            base_url=self.api_base_url,
            api_key="dummy-key"  # Many local APIs don't validate this
        )
        return client

    @property
    def load_tokenizer(self):
        """Local API doesn't provide tokenizer access"""
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
        **kwargs
    ) -> str:
        prompt = data_blob["prompt"]
        max_tokens = max_new_tokens
        generation_config = {
            **kwargs,
            **{"temperature": temperature, "max_tokens": max_tokens},
        }
        print(prompt)
        try:
            completion = (
                self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    **generation_config
                )
                .choices[0]
                .text
            )
            
            return self.postprocess(output_string=completion) if postprocess else completion
            
        except Exception as e:
            from premsql.logger import setup_console_logger
            logger = setup_console_logger(name="[API_GENERATOR]")
            logger.error(f"Error generating SQL with model {self.model_name}: {str(e)}")
            raise