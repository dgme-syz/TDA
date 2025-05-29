import os
import random

from .utils import Datum, DatasetBase, read_json


template = ['a photo of a {}, a type of pet.']

class OxfordPets(DatasetBase):

    dataset_dir = 'oxford_pets'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')

        self.template = [
            lambda c: f"a photo of a {c}, a type of pet.",
        ]

        test = self.read_split(self.split_path, self.image_dir)
        train_x = self.read_split(self.split_path, self.image_dir, split="train")
        super().__init__(test=test, train_x=train_x)
    
    @staticmethod
    def read_split(filepath, path_prefix, split="test"):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split_way = read_json(filepath)
        d = _convert(split_way[split])

        return d
    

