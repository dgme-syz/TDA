import argparse
import torch
from typing import Union, List
from transformers import CLIPModel, CLIPProcessor
from data_modules import AutoDataset
from utils import extract_clip_text_weights, eval


def main(args):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", device_map="auto", torch_dtype="auto")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    torch.manual_seed(42)
    
    
    # Eval
    for dataset_name in args.dataset:
        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, template = AutoDataset(dataset_name)
        clip_weights = extract_clip_text_weights(
            class_label=label, template=template, clip_model=model, tokenizer=processor.tokenizer
        )
        
        eval(
            
        )
        

        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=Union[str, List[str]], required=True, 
        help="Dataset name or list of dataset names"
    )
    
    parser.add_argument(
        "--cache_config", 
        type=str, 
        required=True
    )
    
    
    args = parser.parse_args()
    if isinstance(args.dataset, str):
        args.dataset = [args.dataset]
        
    return args