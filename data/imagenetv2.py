import os
from .utils import Datum, DatasetBase, listdir_nohidden

from .imagenet import ImageNet


class ImageNetV2(DatasetBase):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)
        self.template = [
            lambda c: f"itap of a {c}.",
            lambda c: f"a bad photo of the {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"a photo of the large {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"art of the {c}.",
            lambda c: f"a photo of the small {c}."
        ]
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data) 

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items