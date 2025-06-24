import os
import time
from typing import Any, Dict, Optional
import re

import requests

from utils.custom_generators.base import Text2SQLGeneratorBase

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")


class WikiSQLText2SQLGeneratorAPI(Text2SQLGeneratorBase):
    """
    Generic API generator for Text2SQL benchmarking.
    Works with any local API endpoint that accepts OpenAI-compatible requests.
    No API keys required - designed for local APIs.
    """
    def lowercase_quoted_strings(self, sql_string):
        def replace_quoted(match):
            # Get the full match including quotes
            full_match = match.group(0)
            # Get just the content between quotes
            content = match.group(1)
            # Return with lowercase content
            return f"'{content.lower()}'"
        pattern = r"'([^'\\]*(\\.[^'\\]*)*)'"

        return re.sub(pattern, replace_quoted, sql_string)
    
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
        retries: int = 3,
        retry_delay: float = 2.0,
        use_extended_api: bool = False,  # Choose between OpenAI client vs direct HTTP
        **kwargs
    ) -> str:
        """
        Generate SQL using either standard OpenAI client or direct HTTP calls
        
        Args:
            data_blob: Dictionary containing the prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_new_tokens: Maximum tokens to generate
            postprocess: Whether to postprocess the output
            retries: Number of retry attempts
            retry_delay: Delay between retries
            use_extended_api: If True, use direct HTTP for full parameter support
            **kwargs: Additional generation parameters supported by your server:
                - num_beams: Beam search width (recommended: 4 for SQL)
                - do_sample: Force sampling on/off
                - repetition_penalty: Reduce repetitive text (recommended: 1.1)
                - length_penalty: Length preference for beam search
                - early_stopping: Early stopping for beam search
                - no_repeat_ngram_size: Prevent n-gram repetition
                - stop: List of stop sequences
                - top_p: Nucleus sampling threshold
        """
        prompt = data_blob["prompt"]
        max_tokens = max_new_tokens
        
        # Prepare generation config
        generation_config = {
            **kwargs,
            **{"temperature": temperature, "max_tokens": max_tokens},
        }
        
        attempt = 0
        while attempt < retries:
            try:
                if use_extended_api:
                    # Use direct HTTP requests - supports ALL parameters
                    completion = self._call_direct_api(
                        prompt=prompt,
                        generation_config=generation_config
                    )
                else:
                    # Use standard OpenAI client - limited parameters only
                    completion = self._call_openai_client(
                        prompt=prompt,
                        generation_config=generation_config
                    )
                
                if postprocess:
                    completion = self.postprocess(output_string=completion)
                
                return self.lowercase_quoted_strings(completion)
                
                

            except Exception as e:
                attempt += 1
                print(f"[Attempt {attempt}] Error: {e}")
                from premsql.logger import setup_console_logger
                logger = setup_console_logger(name="[API_GENERATOR]")
                logger.error(f"Error generating SQL with model {self.model_name}: {str(e)}")

                if attempt >= retries:
                    raise  # re-raise after final failed attempt
                time.sleep(retry_delay)

    def _call_openai_client(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Use standard OpenAI client - filters out extended parameters
        Only supports: temperature, max_tokens, top_p, stop, stream
        """
        # Filter to only standard OpenAI parameters
        standard_params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": generation_config.get("temperature", 0.0),
            "max_tokens": generation_config.get("max_tokens", 256),
            "top_p": generation_config.get("top_p"),
            "stop": generation_config.get("stop"),
            "stream": generation_config.get("stream", False)
        }
        
        # Remove None values
        standard_params = {k: v for k, v in standard_params.items() if v is not None}
        
        # Log filtered parameters
        filtered_out = [k for k in generation_config.keys() 
                    if k not in standard_params and k not in ["temperature", "max_tokens"]]
        if filtered_out:
            print(f"⚠️  Standard API: Filtered out extended parameters: {filtered_out}")
        
        completion = (
            self.client.completions.create(**standard_params)
            .choices[0]
            .text
        )
        
        return completion

    def _call_direct_api(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Use direct HTTP requests to your server - supports ALL parameters
        Supports: num_beams, repetition_penalty, do_sample, length_penalty, etc.
        """
        # All parameters are supported by your server
        request_params = {
            "model": self.model_name,
            "prompt": prompt,
            **generation_config  # Include ALL parameters
        }
        
        # Remove None values
        request_params = {k: v for k, v in request_params.items() if v is not None}
        
        # Log extended parameters being used
        extended_params = [k for k in request_params.keys() 
                        if k in {"num_beams", "repetition_penalty", "do_sample", 
                                "length_penalty", "early_stopping", "no_repeat_ngram_size"}]
        if extended_params:
            print(f"🔧 Extended API: Using parameters: {extended_params}")
        
        # Make direct HTTP request to your server's /v1/completions endpoint
        response = requests.post(
            f"{self.api_base_url}/completions",
            json=request_params,
            headers={"Content-Type": "application/json"},
            timeout=120  # Longer timeout for beam search
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        completion = result["choices"][0]["text"]
        
        return completion