import os
from utils.regular_datasets import Text2SQLDataset
from utils.custom_executors_subsets import SQLiteExecutor
from utils.custom_evaluator_subsets import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorAPI


# Initialize the BirdBench Dataset
spider_dataset = Text2SQLDataset(
    dataset_name='spider', 
    split="validation", 
    force_download=False,
    dataset_folder="source/datasets",
).setup_dataset( prompt_template="source/prompts/spider_prompt.md")


# Initialize the OpenRouter generator
generator = Text2SQLGeneratorAPI(
    model_name="chatdb_natural-sql-7b",  # Replace with your model name
    experiment_name="spider_chatdb_natural-sql-7b",
    type="test",
    api_base_url="http://0.0.0.0:8001/v1",  # Using root endpoint, client will append /completions
)

# Initialize executor
executor = SQLiteExecutor()

# Get the responses with execution-guided decoding
responses = generator.generate_and_save_results(
    dataset=spider_dataset,
    temperature=0,
    max_new_tokens=4000,
    force=True,
    postprocess=True,
    executor=executor,
    max_retries=2,
    use_extended_api=True,
    stop=[";", "```"],
)


# Define the evaluator
evaluator = Text2SQLEvaluator(
    executor=executor,
    experiment_path=generator.experiment_path
)

# Now evaluate the models
results = evaluator.execute(
    metric_name="accuracy",
    model_responses=responses,
    meta_time_out=10
)

print("Evaluation Results:")
print(results)

