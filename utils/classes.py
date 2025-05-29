import torch
import os
import json
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
from .augmix import get_ood_preprocess

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items

class ClipImageOutput:
    loss: torch.Tensor
    prob_map: torch.Tensor
    pred: torch.Tensor
    clip_image_logits: torch.Tensor
    image_embeds: torch.Tensor
    
    def __init__(
        self, 
        loss: torch.Tensor,
        prob_map: torch.Tensor,
        pred: torch.Tensor,
        clip_image_logits: torch.Tensor,
        image_embeds: torch.Tensor
    ):
        self.loss = loss
        self.prob_map = prob_map
        self.pred = pred
        self.clip_image_logits = clip_image_logits
        self.image_embeds = image_embeds
        
    def __iter__(self):
        return iter([
            self.loss, 
            self.prob_map, 
            self.pred, 
            self.clip_image_logits, 
            self.image_embeds
        ])

    
class DataWrapper(Dataset):
    def __init__(self, streaming_data, augmix=False):
        self.data = streaming_data
        self.augmenter = get_ood_preprocess(augmix=augmix)
        
    def __getitem__(self, index):
        item = self.data[index]
        if isinstance(item, tuple):
            image, label = item
        else:
            image, label = read_image(item.impath), item.label
        image = self.process(image)
        return {
            "image": image,
            "label": torch.tensor([label], requires_grad=False)
        }
    
    def process(
        self, 
        images, 
        **kwargs
    ) -> torch.Tensor:
        return self.augmenter(images)
        
    def __len__(self):
        return len(self.data)