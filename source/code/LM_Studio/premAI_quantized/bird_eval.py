import os
from utils.regular_datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from utils.custom_evaluator import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorAPI


# Initialize the BirdBench Dataset
spider_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="validation",
    force_download=False,
    dataset_folder="source/datasets",
).setup_dataset(num_rows=10, prompt_template="source/prompts/bird_prompt.md")


# Initialize the OpenRouter generator
generator = Text2SQLGeneratorAPI(
    model_name="premAI_quantized",  # Replace with your model name
    experiment_name="bird_premAI_quantized_LM_Studio",
    type="test",
    api_base_url="http://localhost:1234/v1",  # Using root endpoint, client will append /completions
)

# Initialize executor
executor = SQLiteExecutor()

# Get the responses with execution-guided decoding
responses = generator.generate_and_save_results(
    dataset=spider_dataset,
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

