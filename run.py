import argparse
import torch
from typing import Union, List
from transformers import CLIPModel, CLIPProcessor
from data_modules import AutoDataset
from utils import extract_clip_text_weights, eval, CacheModule


def main(args):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", device_map="auto", torch_dtype="auto")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    cache = CacheModule.load_from_yaml(args.cache_config)
    print(cache)
    torch.manual_seed(42)
    
    # Eval
    for dataset_name in args.dataset:
        torch.cuda.empty_cache()
        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, template = AutoDataset(dataset_name)
        clip_weights = extract_clip_text_weights(
            class_label=label, template=template, clip_model=model, processor=processor
        )
            
        results = eval(
            dataset=data, 
            model=model, 
            clip_weights=clip_weights, 
            cache=cache, 
            label=label, 
            text_embeds=clip_weights, 
            processor=processor
        )
        
        print(
            f"Results for {dataset_name}[ACC]: {results}"
        )

        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        default="food101", 
        help="Dataset name or list of dataset names"
    )
    
    parser.add_argument(
        "--cache_config", 
        default="./configs/food101.yaml", 
        help="Path to cache configuration file"
    )
    
    
    args = parser.parse_args()
    if isinstance(args.dataset, str):
        args.dataset = [args.dataset]
        
    return args

if __name__ == "__main__":
    main(get_args())