import os
from utils.regular_datasets import Text2SQLDataset
from utils.custom_executors_subsets import SQLiteExecutor
from utils.custom_evaluator_subsets import Text2SQLEvaluator
from utils.custom_generators import Text2SQLGeneratorAPI


bird_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="validation",
    force_download=False,
    dataset_folder="source/datasets",
).setup_dataset( prompt_template="source/prompts/bird_prompt.md")


generator = Text2SQLGeneratorAPI(
    model_name="premAI",  # Replace with your model name
    experiment_name="bird_premAI",
    type="test",
    api_base_url="http://0.0.0.0:7860/v1",  # Using root endpoint, client will append /completions
)

executor = SQLiteExecutor()

responses = generator.generate_and_save_results(
    dataset=bird_dataset,
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
    filter_by="difficulty",
    meta_time_out=10
)

print("Evaluation Results:")
print(results)

