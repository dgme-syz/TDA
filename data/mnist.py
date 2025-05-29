import os
import torchvision
from collections import OrderedDict


class MNIST():

    dataset_dir = 'mnist'

    def __init__(self, root):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        
        self.test = torchvision.datasets.MNIST(
            root=self.image_dir,
            train=False,
            download=True,
        )
        self.train_x = torchvision.datasets.MNIST(
            root=self.image_dir,
            train=True,
            download=True,
        )
        self.template = [
            lambda c: f'a photo of the number: "{c}".',
        ]
        self.classnames = self.train_x.classes
    
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames
if __name__ == "__main__":
    x = MNIST("./dataset", None)