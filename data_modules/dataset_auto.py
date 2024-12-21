
from datasets import load_dataset, Dataset, ClassLabel
from collections import OrderedDict
from typing import List, Union
from functools import lru_cache

DATASET_NAME_TO_TEMPLATE = OrderedDict(
    [
        ("food101", ["a photo of {}, a type of food."]),
    ]
)

DATASET_NAME_TO_HUBADDR = OrderedDict(
    [
        ("food101", "ethz/food101"),
    ]
)

class LoadDatasetOutput:
    data: Dataset
    label: ClassLabel
    template: Union[List[str], str]
    
    def __init__(self, data: Dataset, label: ClassLabel, template: Union[List[str], str]):
        self.data = data
        self.label = label
        self.template = template
    def __iter__(self):
        return iter([self.data, self.label, self.template])

def AutoDataset(dataset_name: str) -> LoadDatasetOutput:
    data = load_dataset(DATASET_NAME_TO_HUBADDR[dataset_name])
    # [TODO] need a better way to find labels
    def find_labels(ds: Dataset) -> ClassLabel:
        return ds["train"].info.features["label"]

    label = find_labels(data)
    template = DATASET_NAME_TO_TEMPLATE[dataset_name]
    data = data["validation"]
    
    print(f"Dataset: {dataset_name} Loaded")
    return LoadDatasetOutput(
        data=data,
        label=label,
        template=template
    )