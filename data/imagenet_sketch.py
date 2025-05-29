import os
from .utils import Datum, DatasetBase, listdir_nohidden 

from .imagenet import ImageNet


class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.template = [
            lambda c: f"a bad photo of a {c}.",
            lambda c: f"a photo of many {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a photo of the hard to see {c}.",
            lambda c: f"a low resolution photo of the {c}.",    
            lambda c: f"a rendering of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a bad photo of the {c}.",
            lambda c: f"a cropped photo of the {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"a photo of a hard to see {c}.",
            lambda c: f"a bright photo of a {c}.",
            lambda c: f"a photo of a clean {c}.",
            lambda c: f"a photo of a dirty {c}.",
            lambda c: f"a dark photo of the {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a photo of the cool {c}.",
            lambda c: f"a close-up photo of a {c}.",
            lambda c: f"a black and white photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a pixelated photo of the {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a bright photo of the {c}.",
            lambda c: f"a cropped photo of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a photo of the dirty {c}.",
            lambda c: f"a jpeg corrupted photo of a {c}.",
            lambda c: f"a blurry photo of the {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a good photo of the {c}.",
            lambda c: f"a rendering of the {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a photo of one {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a close-up photo of the {c}.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a doodle of the {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"a low resolution photo of a {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"a photo of the clean {c}.",
            lambda c: f"a photo of a large {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a photo of a nice {c}.",
            lambda c: f"a photo of a weird {c}.",
            lambda c: f"a blurry photo of a {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a pixelated photo of a {c}.",
            lambda c: f"itap of the {c}.",
            lambda c: f"a jpeg corrupted photo of the {c}.",
            lambda c: f"a good photo of a {c}.",
            lambda c: f"a rendering of the {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a photo of the nice {c}.",
            lambda c: f"a photo of the small {c}.",
            lambda c: f"a photo of the weird {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a photo of the large {c}.",
            lambda c: f"a black and white photo of a {c}.",
            lambda c: f"the plushie {c}.",
            lambda c: f"a dark photo of a {c}.",
            lambda c: f"itap of a {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"itap of my {c}.",
            lambda c: f"a photo of a cool {c}.",
            lambda c: f"a photo of a small {c}.",
            lambda c: f"a tattoo of the {c}.",
        ]

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data)  

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
