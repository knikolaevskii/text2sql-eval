class WikiSQLDataset:
    def __init__(self, dataset_name: str, dataset_folder: str):
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder

    def setup_dataset(self):
        return self.dataset_name