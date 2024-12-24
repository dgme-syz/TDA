from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from datasets import ClassLabel
from typing import List, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from .cache_module import CacheModule
from .classes import ClipImageOutput
import re
import numpy as np
from .classes import DataWrapper


wordnet_pattern = re.compile(r"n[0-9]{8}")

def convert_wordnet_to_cls(name: str):
    from nltk.corpus import wordnet
    return wordnet.synset_from_pos_and_offset(
        'n', int(re.search(r"^n(\d{8})$", name).group(1))
    ).name().split('.')[0]


def extract_clip_text_weights(
    class_label: ClassLabel,
    template: Union[List[str], str],
    clip_model: CLIPModel, 
    processor: CLIPProcessor, 
    **kwargs
) -> List[torch.Tensor]:
    with torch.no_grad():
        clip_weights = []
        
        for cls in class_label.names:
            if wordnet_pattern.match(cls) is not None:
                cls = convert_wordnet_to_cls(cls)
            prompts = [t.format(cls) for t in template]
            input_ids = processor(
                text=prompts, return_tensors="pt", padding=True
            )["input_ids"].to(clip_model.device)
            
            cls_embed = clip_model.get_text_features(input_ids) # batch_size x hidden_size
            cls_embed = cls_embed.mean(dim=0) # mean, beaceuse we have multiple templates in one cls
            cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
            clip_weights.append(cls_embed)
        
        clip_weights = torch.stack(clip_weights, dim=1)  
        assert len(clip_weights.shape) == 2
        return clip_weights # hidden_size x num_classes
     
        
def softmax_entropy(x: torch.Tensor):
    # x: batch_size x num_classes
    return -torch.sum(torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1), dim=1)     

def avg_entropy(x: torch.Tensor):
    # x: batch_size x num_classes
    batch_size = x.shape[0]
    logits = x - x.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(batch_size) # 1 x num_classes, 
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1) # 1 x 1

def get_clip_image_info(
    images: torch.Tensor,
    model: CLIPModel,
    text_embeds: torch.Tensor,
    **kwargs
):
    with torch.no_grad():
        batch_size = images.shape[0]
        image_embeds = model.get_image_features(images)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        clip_image_logits = 100. * image_embeds @ text_embeds # batch_size x num_classes
        
        if batch_size == 1:
            loss = softmax_entropy(clip_image_logits)
            prob_map = clip_image_logits.softmax(dim=-1) # batch_size x num_classes, 

            pred = prob_map.argmax(dim=-1)
        else:
            batch_entropy = softmax_entropy(clip_image_logits)
            select_idx = torch.argsort(
                batch_entropy, descending=False
            )[:int(batch_size * 0.1)]
            output = clip_image_logits[select_idx]
            image_embeds = image_embeds[select_idx].mean(0).unsqueeze(0)
            clip_image_logits = output.mean(0).unsqueeze(0)
            
            loss = avg_entropy(output)
            prob_map = output.softmax(dim=-1).mean(0).unsqueeze(0) # 1 x num_classes
            pred = prob_map.argmax(dim=-1) # 1, 

        return ClipImageOutput(
            loss=loss,
            prob_map=prob_map,
            pred=pred,
            clip_image_logits=clip_image_logits,
            image_embeds=image_embeds, 
        )
    

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
    print(f"Number of classes: {num_classes}")
    acc = {
        "correct": 0, "total": 0
    }
    
    data_loader = DataLoader(
        dataset=DataWrapper(dataset, augmix=True), 
        batch_size=1, 
        shuffle=True, 
        pin_memory=torch.cuda.is_available(), 
        num_workers=2 # -> 2 进程 4 线程
    )
    
    pbar = tqdm(data_loader, desc="Processed test images: ") 
    with torch.no_grad():
        for x in pbar:
            # torch.cuda.empty_cache()
            images, target = x["image"], x["label"]
            images = torch.cat(images, dim=0).to(model.device)
            target = target.to(model.device)
            
            loss, prob_map, pred, clip_image_logits, image_embeds = get_clip_image_info(
                images=images, 
                model=model, 
                text_embeds=text_embeds, 
                **kwargs
            )
            prop_entropy = float((loss / num_classes).item())
            
            cache.update_pos_cache(pred, image_embeds, loss)
            cache.update_neg_cache(pred, image_embeds, loss, True, prob_map, prop_entropy)
            
            final_logits: torch.Tensor = clip_image_logits.clone()
            if cache.pos_enabled:
                cache_pos_logits = cache.compute_pos_logits(image_embeds, num_classes)
            if cache.neg_enabled:
                cache_neg_logits = cache.compute_neg_logits(image_embeds)
            
            if cache_pos_logits is not None: final_logits += cache_pos_logits
            if cache_neg_logits is not None: final_logits -= cache_neg_logits
            
            union_pred = final_logits.argmax(dim=-1)
            acc["correct"] += torch.sum(union_pred == target).item()
            acc["total"] += len(target)
            # print(f"Accuracy: {acc['correct'] / acc['total']}")
            accuracy = acc["correct"] / acc["total"] * 100
            pbar.set_postfix({
                "acc": f"{accuracy:.2f}%" 
            })

    return acc["correct"] / acc["total"]
        
        
        
        

    