from transformers import CLIPModel, CLIPProcessor
from datasets import ClassLabel
from typing import List, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from .cache_module import CacheModule
from .classes import ClipImageOutput

@torch.inference_mode()
def extract_clip_text_weights(
    class_label: ClassLabel,
    template: Union[List[str], str],
    clip_model: CLIPModel, 
    processor: CLIPProcessor, 
    **kwargs
) -> List[torch.Tensor]:
    
    clip_weights = []
    
    for cls in class_label.names:
        prompts = [t.format(cls) for t in template]
        input_ids = processor(
            text=prompts, return_tensors="pt"
        )["input_ids"].to(clip_model.device)
        
        cls_embed = clip_model.get_text_features(input_ids) # batch_size x hidden_size
        cls_embed = cls_embed.mean(dim=0) # mean, beaceuse we have multiple templates in one cls
        cls_embed = cls_embed / cls_embed.norm()
        clip_weights.append(cls_embed)
    
    clip_weights = torch.stack(clip_weights, dim=1)  
    assert len(clip_weights.shape) == 2
    return clip_weights # hidden_size x num_classes
     
def preprocess(info: dict, image_decoder: CLIPProcessor):
    info["image"] = image_decoder(info["image"])
        
def softmax_entropy(x: torch.Tensor):
    # x: batch_size x num_classes
    x = torch.softmax(x, dim=1)
    return -torch.sum(x * torch.log(x), dim=1)     

def get_clip_image_info(
    images: torch.Tensor, 
    model: CLIPModel,
    text_embeds: torch.Tensor,
    **kwargs
):
    assert len(images.shape) == 4, \
        "Images shape expected to be batch_size x 3 x H x W"
    if images.shape[0] != 1:
        Warning.warn(
            "Batch size is not 1, this is not recommended for inference"
        )
    
    image_embeds = model.get_image_features(images.to(model.device))
    image_embeds = image_embeds / image_embeds.norm()

    logit_scale = model.logit_scale.exp()
    clip_image_logits = logit_scale * image_embeds @ text_embeds # batch_size x num_classes
    
    loss = softmax_entropy(clip_image_logits)
    prob_map = clip_image_logits.softmax(dim=-1) # batch_size x num_classes, 

    assert len(prob_map.shape) == 2, "Prob map shape mismatch"
    pred = prob_map.argmax(dim=-1)

    return ClipImageOutput(
        loss=loss,
        prob_map=prob_map,
        pred=pred,
        clip_image_logits=clip_image_logits,
        image_embeds=image_embeds, 
    )
    

    
@torch.inference_mode()
def eval(
    dataset: Dataset,
    model: CLIPModel,
    text_embeds: torch.Tensor,
    cache: CacheModule,
    label: ClassLabel,
    processor: CLIPProcessor,
    **kwargs
):
    num_classes = len(label.names)
    acc = {
        "coorect": 0, "total": 0
    }
    for x in tqdm(dataset, desc="Processed test images: "):
        images, target = x["image"], x["label"]
        
        if isinstance(target, int):
            images = [images]
            target = [target]
            
        images = processor(images=images, return_tensors="pt")["pixel_values"]
        images = images.to(model.device)
        # print(
        #     f"Shape of images: {type(images[0][0].shape)}, target: {type(target)}"
        # )
        
        target = torch.tensor(target)
        target = target.to(model.device)
        loss, prob_map, pred, clip_image_logits, image_embeds = get_clip_image_info(
            images=images, model=model, text_embeds=text_embeds, **kwargs
        )

        prop_entropy = float((loss / num_classes).item())
        
        cache.update_pos_cache(pred, image_embeds, loss)
        cache.update_neg_cache(pred, image_embeds, loss, True, prob_map, prop_entropy)
        
        final_logits: torch.Tensor = clip_image_logits.clone()
        cache_pos_logits = cache.compute_pos_logits(image_embeds, num_classes)
        cache_neg_logits = cache.compute_neg_logits(image_embeds)
        
        if cache_pos_logits is not None: final_logits += cache_pos_logits
        if cache_neg_logits is not None: final_logits -= cache_neg_logits
        
        acc["coorect"] += torch.sum(pred == target).item()
        acc["total"] += len(target)
        # print(f"Accuracy: {acc['coorect'] / acc['total']}")
    return acc["coorect"] / acc["total"]
        
        
        
        

    