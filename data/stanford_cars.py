import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = [
            lambda c: f"a photo of a {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}.",
        ]

        test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        train_x = OxfordPets.read_split(self.split_path, self.dataset_dir, split="train")
        super().__init__(test=test, train_x=train_x)