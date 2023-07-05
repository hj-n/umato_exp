import os
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'npy')

dataset_names = []

for filename in os.listdir(DATASET_PATH):
    filepath = os.path.join(DATASET_PATH, filename)
    if os.path.isdir(filepath):
        dataset_names.append(filename)

dataset_names.sort()


class Dataset:
    def __init__(self, dataset_name):
        self.name = dataset_name
        dirpath = os.path.join(DATASET_PATH, self.name)
        assert os.path.isdir(dirpath)
        data_path = os.path.join(dirpath, 'data.npy')
        label_path = os.path.join(dirpath, 'label.npy')
        assert os.path.isfile(data_path)
        assert os.path.isfile(label_path)
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        assert len(self.data) == len(self.label)

    def __repr__(self):
        return f"[Dataset | {self.name}: {len(self.data)}]"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def list_datasets():
    return dataset_names


def load_dataset(dataset_name):
    return Dataset(dataset_name)


def load_all_datasets():
    all_datasets = []
    for dataset_name in list_datasets():
        all_datasets.append(load_dataset(dataset_name))
    return all_datasets
