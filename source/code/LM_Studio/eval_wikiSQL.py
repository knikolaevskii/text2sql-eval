from utils.custom_generators import Text2SQLGeneratorAPI
from utils.custom_datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

wiki_dataset = Text2SQLDataset(
    dataset_name='wikisql',
    split="test",
    dataset_folder="source/datasets",
    prompt_template="source/prompts/new_prompt.md",
    data_schema_file="source/datasets/wikisql/test_wiki_sql_metadata.json"
).setup_dataset(num_rows=10, custom_db_path="source/datasets/wikisql/database/test.db")

# Initialize the API generator with your local endpoint
generator = Text2SQLGeneratorAPI(
    model_name="some",  # Replace with your model name
    experiment_name="test_api_generators",
    type="test",
    api_base_url="http://localhost:1234/v1",  # Using root endpoint, client will append /chat/completions
)

# Get the responses
responses = generator.generate_and_save_results(
    dataset=wiki_dataset,
    temperature=0.1,
    max_new_tokens=256,
    force=True,
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
    filter_by="db_id",
    meta_time_out=10
)

print("Evaluation results:")
print(results)
