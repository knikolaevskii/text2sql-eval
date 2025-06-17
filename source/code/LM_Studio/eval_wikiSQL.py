from premsql.generators import Text2SQLGeneratorAPI
from utils.custom_datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

wiki_dataset = Text2SQLDataset(
    dataset_name='wikisql',
    split="test",
    dataset_folder="source/datasets",
    prompt_template="source/prompts/new_prompt.md",
    data_schema_file="source/datasets/wikisql/test_wiki_sql_metadata.json"  # Add this line
).setup_dataset(num_rows=1)
