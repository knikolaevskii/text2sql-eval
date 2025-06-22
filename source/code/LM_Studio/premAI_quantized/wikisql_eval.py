from utils.custom_datasets_wikisql import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from utils.custom_evaluator import Text2SQLEvaluator
from utils.custom_generators import WikiSQLText2SQLGeneratorAPI


wiki_dataset = Text2SQLDataset(
    dataset_name='wikisql',
    split="test",
    dataset_folder="source/datasets",
    prompt_template="source/prompts/wikisql_prompt.md",
    data_schema_file="source/datasets/wikisql/test_wiki_sql_metadata.json"
).setup_dataset(num_rows=10, custom_db_path="source/datasets/wikisql/database/test.db")

# Initialize the OpenRouter generator
generator = WikiSQLText2SQLGeneratorAPI(
    model_name="premAI_quantized",  # Replace with your model name
    experiment_name="wikisql_premAI_quantized_LM_Studio",
    type="test",
    api_base_url="http://localhost:1234/v1",  # Using root endpoint, client will append /completions
)

# Initialize executor
executor = SQLiteExecutor()

# Get the responses with execution-guided decoding
responses = generator.generate_and_save_results(
    dataset=wiki_dataset,
    temperature=0,
    max_new_tokens=256,
    force=True,
    postprocess=True,
    executor=executor,
    max_retries=3
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
