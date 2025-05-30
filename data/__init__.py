from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .cifar100 import Cifar100
from .mnist import MNIST

dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "imagenet-a": ImageNetA,
    "imagenet-v": ImageNetV2,
    "imagenet-r": ImageNetR,
    "imagenet-s": ImageNetSketch,
    "cifar100": Cifar100,
    "mnist": MNIST,
}


def build_dataset(dataset, root_path=None, eval=True):
    if root_path is None:
        root_path = "./dataset"
    data = dataset_list[dataset](root_path)
    
    outs = (data.test, ) if eval else (data.train_x, )
    outs += (data.classnames, data.template[0], )
    return outs

