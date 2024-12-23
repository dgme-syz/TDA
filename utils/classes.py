import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from concurrent.futures import ProcessPoolExecutor
from transformers import CLIPProcessor
from .augmix import get_ood_preprocess
from torchvision import transforms

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
        image, label = item["image"], item["label"]
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