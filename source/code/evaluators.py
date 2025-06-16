from premsql.generators import Text2SQLGeneratorHF
from premsql.datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

# Initialize the BirdBench Dataset
bird_dataset = Text2SQLDataset(
    dataset_name='bird', split="validation", force_download=False,
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets"
).setup_dataset(num_rows=10)

# Initialize the generator
generator = Text2SQLGeneratorHF(
    model_or_name_or_path="premai-io/prem-1B-SQL",
    experiment_name="test_generators",
    device="cpu",
    type="test"
)

# Get the responses
responses = generator.generate_and_save_results(
    dataset=bird_dataset,
    temperature=0.1,
    max_new_tokens=256,
    force=True,
)

print(responses)
# response = generator.generate_and_save(
#     dataset=bird_dataset,
#     temperature=0.1,
#     max_new_tokens=256,
#     force=True
# )


# Define the executor
executor = SQLiteExecutor()

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

print(results)

