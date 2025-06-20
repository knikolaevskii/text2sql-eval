from utils.custom_generators import Text2SQLGeneratorAPI
from utils.custom_datasets_wikisql import Text2SQLDataset
from premsql.executors import SQLiteExecutor
from premsql.evaluator import Text2SQLEvaluator

import json
import re
from pathlib import Path

def lowercase_quoted_strings(sql_string):
    def replace_quoted(match):
        # Get the full match including quotes
        full_match = match.group(0)
        # Get just the content between quotes
        content = match.group(1)
        # Return with lowercase content
        return f"'{content.lower()}'"
    pattern = r"'([^'\\]*(\\.[^'\\]*)*)'"
    
    return re.sub(pattern, replace_quoted, sql_string)

def process_json_file(file_path):
 
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
    
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded JSON file with {len(data)} entries")
        
        # Process each entry
        modified_count = 0
        for i, entry in enumerate(data):
            if 'generated' in entry and isinstance(entry['generated'], str):
                original_sql = entry['generated']
                modified_sql = lowercase_quoted_strings(original_sql)
                
                if original_sql != modified_sql:
                    entry['generated'] = modified_sql
                    modified_count += 1
                    print(f"Entry {i+1}:")
                    print(f"  Original: {original_sql}")
                    print(f"  Modified: {modified_sql}")
                    print()
        
        print(f"Modified {modified_count} entries out of {len(data)} total entries")
        
        if modified_count > 0:
            # Create backup
            backup_path = file_path.with_suffix('.backup.json')
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Created backup at: {backup_path}")
            
            # Save the modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Updated original file: {file_path}")
        else:
            print("No modifications needed.")
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
    except Exception as e:
        print(f"Error processing file: {e}")


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
    api_base_url="http://localhost:1234/v1",  # Using root endpoint, client will append /completions
)

# Get the responses
responses = generator.generate_and_save_results(
    dataset=wiki_dataset,
    temperature=0,
    max_new_tokens=256,
    force=False,
    postprocess=True
)

process_json_file("/Users/kirillnikolaevskii/Library/Caches/premsql/experiments/test/test_openrouter_generators/predict.json")

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
