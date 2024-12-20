from transformers import CLIPModel, CLIPTokenizer, CLIPTokenizerFast
from datasets import ClassLabel
from typing import List, Union
import torch
from datasets import Dataset


@torch.no_grad()
def extract_clip_text_weights(
    class_label: ClassLabel,
    template: Union[List[str], str],
    clip_model: CLIPModel, 
    tokenizer: Union[CLIPTokenizer, CLIPTokenizerFast]
) -> List[torch.Tensor]:
    
    clip_weights = []
    
    for cls in class_label.names:
        prompts = [t.format(cls) for t in template]
        input_ids = tokenizer.tokenize(
            prompts, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(clip_model.device)
        
        cls_embed = clip_model.get_text_features(input_ids) # batch_size x hidden_size
        cls_embed = cls_embed.mean(dim=0) # mean, beaceuse we have multiple templates in one cls
        cls_embed = cls_embed / cls_embed.norm()
        clip_weights.append(cls_embed)
    
    clip_weights = torch.stack(clip_weights, dim=1)  
    assert len(class_label) == 2
    return clip_weights # hidden_size x num_classes
        
        
def eval(
    dataset: Dataset,
    model: CLIPModel,
    clip_weightsL: torch.Tensor,
    cache_config: dict,
    **kwargs
):
    raise NotImplementedError
    