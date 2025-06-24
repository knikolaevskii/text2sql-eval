from utils.custom_datasets_defog import Text2SQLDataset
from utils.custom_executors import SQLiteExecutor
from utils.custom_evaluator import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorAPI


# Initialize the BirdBench Dataset
bird_dataset = Text2SQLDataset(
    dataset_name='defog', 
    split="questions_gen", 
    dataset_folder="source/datasets",
).setup_dataset(num_rows=20, prompt_template="source/prompts/defog_prompt.md")


# Initialize the OpenRouter generator
generator = Text2SQLGeneratorAPI(
    model_name="defog_sqlcoder_7b_2",  # Replace with your model name
    experiment_name="defog_sqlcoder_7b_2",
    type="test",
    api_base_url="http://0.0.0.0:8000/v1",  # Using root endpoint, client will append /completions
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

# print(f"Generated {len(responses)} responses")

# Define the evaluator
evaluator = Text2SQLEvaluator(
    executor=executor,
    experiment_path=generator.experiment_path
)

# Now evaluate the models
results = evaluator.execute(
    metric_name="accuracy",
    model_responses=responses,
    meta_time_out=10,
)

print("Evaluation Results:")
print(results)
