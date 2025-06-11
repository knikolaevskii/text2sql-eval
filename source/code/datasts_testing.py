from premsql.datasets import Text2SQLDataset
from premsql.utils import print_data

# bird_dataset = Text2SQLDataset(
#     dataset_name='bird', split="train", force_download=False,
#     dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing" # change this to the path where you want to store the dataset
# )

# # raw_bird_training_dataset = bird_dataset.raw_dataset
# # print(raw_bird_training_dataset[0])

# # Set up the BirdBench dataset
# bird_dataset = bird_dataset.setup_dataset(
#     model_name_or_path="premai-io/prem-1B-SQL",
#     num_fewshot=3,
#     num_rows=3
# )

# print(print_data(bird_dataset[0]))

# ------------------------------------------------------------

# bird_dataset_without_tokenization = Text2SQLDataset(
#     dataset_name='bird', split="train", force_download=False,
#     dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing"
# ).setup_dataset(
#     model_name_or_path=None, num_fewshot=3, num_rows=3
# )

# print(print_data(bird_dataset_without_tokenization[0]))

# ------------------------------------------------------------

bird_dataset = Text2SQLDataset(
    dataset_name='bird', split="validation", force_download=False,
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing"
).setup_dataset(
    model_name_or_path=None,
    num_fewshot=3,
    num_rows=100,
    filter_by=("db_id", "california_schools")  # Filter by specific db_id
)

# Count the number of examples for this db_id
db_count = len([example for example in bird_dataset if example["db_id"] == "california_schools"])
print(f"Number of examples for california_schools: {db_count}")


# Loading Spider Dataset
spider_dataset = Text2SQLDataset(
    dataset_name="spider",
    split="train",
    dataset_folder="/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing",
).setup_dataset(
    num_fewshot=3,
    num_rows=3
)

merged_dataset = [*bird_dataset, *spider_dataset]

print(f"Length of bird dataset: {len(bird_dataset)}")
print(f"Length of spider dataset: {len(spider_dataset)}")
print(f"Length of merged dataset: {len(merged_dataset)}")

print(print_data(merged_dataset[0]))

print(spider_dataset[0]["prompt"])