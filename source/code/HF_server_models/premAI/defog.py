from utils.custom_datasets_defog import Text2SQLDataset
from utils.custom_executors_subsets import SQLiteExecutor
from utils.custom_evaluator_subsets import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorAPI


defog_dataset = Text2SQLDataset(
    dataset_name='defog', 
    split="questions_gen", 
    dataset_folder="source/datasets",
).setup_dataset(prompt_template="source/prompts/defog_prompt.md")


generator = Text2SQLGeneratorAPI(
    model_name="premAI",  # Replace with your model name
    experiment_name="defog_premAI",
    type="test",
    api_base_url="http://0.0.0.0:7860/v1",  # Using root endpoint, client will append /completions
)

executor = SQLiteExecutor()

responses = generator.generate_and_save_results(
    dataset=defog_dataset,
    temperature=0,
    max_new_tokens=4000,
    force=True,
    postprocess=True,
    executor=executor,
    max_retries=2,
    use_extended_api=True,
    stop=[";", "```"],
)


evaluator = Text2SQLEvaluator(
    executor=executor,
    experiment_path=generator.experiment_path
)

results = evaluator.execute(
    metric_name="accuracy",
    model_responses=responses,
    meta_time_out=10,
)

print("Evaluation Results:")
print(results)
