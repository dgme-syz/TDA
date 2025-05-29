import os
from .utils import Datum, DatasetBase, listdir_nohidden

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]

class ImageNetA(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-adversarial"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-a")
        self.template = [
            lambda c: f"itap of a {c}.",
            lambda c: f"a bad photo of the {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"a photo of the large {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"art of the {c}.",
            lambda c: f"a photo of the small {c}.",
        ]

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data) 

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items