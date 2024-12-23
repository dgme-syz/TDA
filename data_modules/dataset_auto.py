
from datasets import load_dataset, Dataset, ClassLabel
from collections import OrderedDict
from typing import List, Union
from functools import lru_cache
import ast

GENERAL_TRMP = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}."
]

DATASET_NAME_TO_TEMPLATE = OrderedDict(
    [
        ("food101", ["a photo of {}, a type of food."]),
        ("imagenet_a", GENERAL_TRMP),
        ("imagenet_hard", GENERAL_TRMP)
    ]
)

DATASET_NAME_TO_HUBADDR = OrderedDict(
    [
        ("food101", "ethz/food101"),
        ("imagenet_a", "barkermrl/imagenet-a"),
        ("imagenet_hard", "taesiri/imagenet-hard"),
    ]
)

CHOOSE_SPLIT = OrderedDict(
    [
        ("food101", "validation"),
        ("imagenet_a", "train"),
        ("imagenet_hard", "validation")
    ]
)

CLASSES_FILE = OrderedDict(
    [
        ("food101", "./total_classes/food101.txt"),
        ("imagenet_a", "./total_classes/imagenet_a.txt"),
        ("imagenet_hard", "./total_classes/imagenet_hard.txt")
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

def find_labels(file_path: str) -> ClassLabel:
    with open(file_path, "r") as f:
        classes = f.read()
    na = ast.literal_eval(classes)
    return ClassLabel(names=na, num_classes=len(na))

def AutoDataset(dataset_name: str) -> LoadDatasetOutput:
    data = load_dataset(DATASET_NAME_TO_HUBADDR[dataset_name])
    label = find_labels(CLASSES_FILE[dataset_name])
    template = DATASET_NAME_TO_TEMPLATE[dataset_name]
    data = data[CHOOSE_SPLIT[dataset_name]]
    
    print(f"Dataset: {dataset_name} Loaded")
    return LoadDatasetOutput(
        data=data,
        label=label,
        template=template
    )