import os
from premsql.datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from utils.custom_evaluator import Text2SQLEvaluator

# Import your custom OpenRouter generator
# from premsql.generators import Text2SQLGeneratorOpenRouter
from utils.custom_generators import Text2SQLGeneratorOpenRouter

# Set your OpenRouter API key (or set as environment variable OPENROUTER_API_KEY)
# os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

# Initialize the BirdBench Dataset
bird_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="validation", 
    force_download=False,
    dataset_folder="source/datasets"
).setup_dataset(num_rows=10)


# Initialize the OpenRouter generator
generator = Text2SQLGeneratorOpenRouter(
    model_name="gpt-4o-mini",  # You can use any model from the mapping or full OpenRouter model ID
    experiment_name="test_openrouter_generators",
    type="test",
    openrouter_api_key="***REMOVED***",  # Will use OPENROUTER_API_KEY env var if None
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
    max_retries=3
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
    filter_by="db_id",
    meta_time_out=10
)

print("Evaluation Results:")
print(results)

# Optional: Print some example responses
# print("\nSample responses:")
# for i, response in enumerate(responses[:3]):  # Show first 3 responses
#     print(f"\nExample {i+1}:")
#     print(f"Question: {response.get('question', 'N/A')}")
#     print(f"Generated SQL: {response.get('generated', 'N/A')}")
#     if 'evidence' in response:
#         print(f"Evidence: {response['evidence']}")