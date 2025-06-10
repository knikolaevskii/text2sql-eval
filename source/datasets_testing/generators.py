from premsql.generators import Text2SQLGeneratorHF
from premsql.datasets import Text2SQLDataset

bird_dataset = Text2SQLDataset(
    dataset_name='bird', 
    split="train", 
    force_download=False,
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing"
)

# Setup the dataset first - this is the missing step!
bird_dataset = bird_dataset.setup_dataset(
    model_name_or_path="premai-io/prem-1B-SQL",
    num_rows=3    # Optional: limit rows for testing
)

generator = Text2SQLGeneratorHF(
    model_or_name_or_path="premai-io/prem-1B-SQL",
    experiment_name="test_generators",
    device="cpu",  # Changed from cuda:0
    type="test"
)

response = generator.generate_and_save(
    dataset=bird_dataset,
    temperature=0.1,
    max_new_tokens=256,
    force=True
)

print(response)