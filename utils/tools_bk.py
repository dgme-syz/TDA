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
import torch.nn.functional as F
import operator

wordnet_pattern = re.compile(r"n[0-9]{8}")

def convert_wordnet_to_cls(name: str):
    from nltk.corpus import wordnet
    return wordnet.synset_from_pos_and_offset(
        'n', int(re.search(r"^n(\d{8})$", name).group(1))
    ).name().split('.')[0]


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
        if wordnet_pattern.match(cls) is not None:
            cls = convert_wordnet_to_cls(cls)
        prompts = [t.format(cls) for t in template]
        input_ids = processor(
            text=prompts, return_tensors="pt", padding=True
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
    return -torch.sum(torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1), dim=1)     

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
    

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().float()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().float()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


    
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
    dataset = dataset.select(range(2))
    num_classes = len(label.names)
    print(f"Number of classes: {num_classes}")
    acc = {
        "correct": 0, "total": 0
    }
    pos_params, neg_params = cache.pos_params, cache.neg_params
    pos_enabled = pos_params["enabled"]
    neg_enabled = neg_params["enabled"]
    pos_cache = {}
    neg_cache = {}
    pbar = tqdm(dataset, desc="Processed test images: ") 
    for x in pbar:
        torch.cuda.empty_cache()
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
        
        # cache.update_pos_cache(pred, image_embeds, loss)
        # cache.update_neg_cache(pred, image_embeds, loss, True, prob_map, prop_entropy)
        
        # final_logits: torch.Tensor = clip_image_logits.clone()
        # cache_pos_logits = cache.compute_pos_logits(image_embeds, num_classes)
        # cache_neg_logits = cache.compute_neg_logits(image_embeds)
        
        # if cache_pos_logits is not None: final_logits += cache_pos_logits
        # if cache_neg_logits is not None: final_logits -= cache_neg_logits
        
        if pos_enabled:
            update_cache(pos_cache, pred, [image_embeds, loss], pos_params['shot_capacity'])

        if neg_enabled:
            update_cache(neg_cache, pred, [image_embeds, loss, prob_map], neg_params['shot_capacity'], True)

        final_logits = clip_image_logits.clone()
        if pos_enabled and pos_cache:
            pos_logits = compute_cache_logits(image_embeds, pos_cache, pos_params['alpha'], pos_params['beta'], text_embeds)
            print(f"Pos logits: {pos_logits}")
            final_logits += pos_logits
        if neg_enabled and neg_cache:
            neg_logits = compute_cache_logits(image_embeds, neg_cache, neg_params['alpha'], neg_params['beta'], text_embeds, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
            print(f"Neg logits: {neg_logits}")
            final_logits -= neg_logits
            
        print(
            f"Final logits: {final_logits}"
            f"pos_logits: {pos_logits}"
            f"neg_logits: {neg_logits}"
        )
        
        
        union_pred = final_logits.argmax(dim=-1)
        acc["correct"] += torch.sum(union_pred == target).item()
        acc["total"] += len(target)
        # print(f"Accuracy: {acc['correct'] / acc['total']}")
        pbar.set_postfix(
            {
                "acc": acc["correct"] / acc["total"],
            }
        )

    return acc["correct"] / acc["total"]
        
        
        
        

    