import os
from utils.regular_datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from utils.custom_evaluator import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorOpenRouter


# Initialize the BirdBench Dataset
bird_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="validation", 
    force_download=False,
    dataset_folder="source/datasets",
).setup_dataset(num_rows=2, prompt_template="source/prompts/bird_prompt.md")


# Initialize the OpenRouter generator
generator = Text2SQLGeneratorOpenRouter(
    model_name="gpt-4o-mini",  # You can use any model from the mapping or full OpenRouter model ID
    experiment_name="bird_gpt_o4_mini_openrouter_generators",
    type="test",
    openrouter_api_key="***REMOVED***",  # Will use OPENROUTER_API_KEY env var if None
    data_base_type="sqlite" # or postgresql
)

# Initialize executor
executor = SQLiteExecutor()

# Get the responses with execution-guided decoding
responses = generator.generate_and_save_results(
    dataset=bird_dataset,
    temperature=0,
    max_new_tokens=256,
    force=True,
    postprocess=True,
    executor=executor,
    max_retries=3,
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
    filter_by="db_id",
    meta_time_out=10
)

print("Evaluation Results:")
print(results)