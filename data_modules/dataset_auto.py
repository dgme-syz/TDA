
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

def AutoDataset(dataset_name):
    data = load_dataset(DATASET_NAME_TO_HUBADDR[dataset_name])

    @lru_cache()
    def find_labels(dataset: Dataset) -> ClassLabel:
        if dataset.hasattr("info"):
            if dataset.info.hasattr("label"):
                return dataset.info.label
            elif dataset.info.hasattr("features"):
                if dataset.info.features.hasattr("label"):
                    return dataset.info.features.label
            else:
                raise NotImplementedError
        else: 
            if dataset.hasattr("train"):
                return find_labels(dataset["train"])
            elif dataset.hasattr("validation"):
                return find_labels(dataset["validation"])
            elif dataset.hasattr("test"):
                return find_labels(dataset["test"])
            elif dataset.hasattr("val"):
                return find_labels(dataset["val"])
            else:
                raise NotImplementedError

    label = find_labels(data)
    template = DATASET_NAME_TO_TEMPLATE[dataset_name]
    
    return LoadDatasetOutput(
        data=data,
        label=label,
        template=template
    )