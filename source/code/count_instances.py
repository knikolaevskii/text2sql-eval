import json
import os

def count_instances(dataset_path):
    total_count = 0
    
    # Count instances in train set
    train_path = "/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing/spider/train.json"
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            train_data = json.load(f)
            print(len(train_data))
            total_count += len(train_data)
    
    # Count instances in validation set
    validation_path = "/Users/kirillnikolaevskii/Desktop/prem/source/datasets_testing/spider/validation.json"
    if os.path.exists(validation_path):
        with open(validation_path, 'r') as f:
            validation_data = json.load(f)
            print(len(validation_data))
            total_count += len(validation_data)
    
    return total_count

if __name__ == "__main__":
    dataset_path = os.path.join('source', 'datasets_testing', 'bird')
    total_instances = count_instances(dataset_path)
    print(f"Total number of instances: {total_instances}") 