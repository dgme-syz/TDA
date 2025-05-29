import os

from .utils import DatasetBase

from .oxford_pets import OxfordPets



class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = [
            lambda c: f"a photo of a person doing {c}.",
        ]

        test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_x = OxfordPets.read_split(self.split_path, self.image_dir, split="train")
        super().__init__(test=test, train_x=train_x)
