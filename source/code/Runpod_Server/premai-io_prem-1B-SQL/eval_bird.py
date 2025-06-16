from premsql.generators import Text2SQLGeneratorAPI  # Assuming you save the class in this location
from premsql.datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

# Initialize the BirdBench Dataset
bird_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="validation",
    force_download=False,
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets"
).setup_dataset(num_rows=1534)

# Initialize the API generator with your local endpoint
generator = Text2SQLGeneratorAPI(
    model_name="premai-io/prem-1B-SQL",  # Use the actual HuggingFace model name
    experiment_name="prem-1B-SQL-test",  # Updated experiment name to match the model
    type="test",
    api_base_url="http://127.0.0.1:8000/v1"  # Using root endpoint, client will append /chat/completions
)

# Get the responses with optimized parameters for this model
responses = generator.generate_and_save_results(
    dataset=bird_dataset,
    temperature=0.0,  # Low temperature for more deterministic SQL generation
    max_new_tokens=256,
    force=False
)

# Define the executor
executor = SQLiteExecutor()

# Define the evaluator
evaluator = Text2SQLEvaluator(
    executor=executor,
    experiment_path=generator.experiment_path
)

# Now evaluate the model results
results = evaluator.execute(
    metric_name="accuracy",
    model_responses=responses,
    filter_by="difficulty",
    meta_time_out=10
)

print("Evaluation results:")
print(results)