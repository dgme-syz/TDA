import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from utils.tools import eval_all
import os
from peft import PeftModel, PeftConfig, get_peft_model
from utils.lora_moe import LoraMoEConfig
import random
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 42
def main(args):
    template = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    model = PeftModel.from_pretrained(model, args.resume).to(torch.bfloat16)
    print(model)

    for name, param in model.named_modules():
        if hasattr(param, "loramoe_router"):
            param.close_task()
            param.set_task_id(args.task_id)
    # Eval
    eval_all(
        model=model, 
        processor=processor, 
        datasets=args.dataset, 
        use_cache=False,
        template=template
    )

        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        default="imagenet_a", 
        help="Dataset name or list of dataset names"
    )
    
    parser.add_argument(
        "--cache_config", 
        default="./configs", 
        help="Path to cache configuration file"
    )
    
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        help="Task ID for the lora",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to the pretrained model"
    )
    
    args = parser.parse_args()
    args.dataset = args.dataset.split(",")
        
    return args

if __name__ == "__main__":
    main(get_args())