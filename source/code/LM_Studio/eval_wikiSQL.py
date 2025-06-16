from premsql.generators import Text2SQLGeneratorAPI  # Assuming you save the class in this location
from premsql.datasets import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

# Initialize the BirdBench Dataset
spider_dataset = Text2SQLDataset(
    dataset_name='spider', 
    split="validation",
    force_download=False,
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets"
).setup_dataset(num_rows=100)

