from premsql.generators import Text2SQLGeneratorAPI  # Assuming you save the class in this location
from utils.custom_datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

wiki_dataset = Text2SQLDataset(
    dataset_name='wikisql', 
    split="test",
    dataset_folder="source/datasets",
    prompt_template = "source/promts/new_prompt.md"
).setup_dataset(num_rows=100)
