import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from data import build_dataset
from utils.tools import extract_clip_text_weights, train
from utils import CacheModule
from peft import LoraConfig, get_peft_model
from utils.lora_moe import LoraMoEConfig
from peft import PeftModel, PeftConfig
import os
import wandb
import random
import numpy as np
from utils.combine_datasets import CombDataset
from safetensors.torch import load_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# python3 train.py --dataset fgvc,caltech101,dtd,eurosat,oxford_flowers,oxford_pets
seed = 42
def main(args):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    target = ["q_proj", "k_proj", "v_proj"]
    # LoRA
    if args.resume == "":
        config = LoraMoEConfig(
            r=32,
            top_k=2, 
            num_tasks=len(args.dataset),
            lora_alpha=16,
            target_modules=target,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights="pissa"
        )
        model = get_peft_model(model, config).to(torch.bfloat16)
    else:
        config = PeftConfig.from_pretrained(args.resume)
        if config.num_tasks != len(args.dataset):
            config.num_tasks = len(args.dataset)
        state_dict = load_file(os.path.join(args.resume, "adapter_model.safetensors"))
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'loramoe_router' not in k
        }
        model = get_peft_model(model, config)
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(torch.bfloat16)

    print(model)
    print(config)
    run = wandb.init(project="CLIP-MOE", config=None, name="exp")

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    torch.manual_seed(seed)
    eval_datasets = []
    template = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]
    

    # Stage1. Train LoRA
    combdataset = CombDataset()
    for i in range(len(args.dataset)):
        for name, param in model.named_modules():
            if hasattr(param, "loramoe_router"):
                param.enable_task(i)
                param.change_router_state(activate=False)
        model.print_trainable_parameters()

        dataset_name = args.dataset[i]
        if dataset_name not in args.train_dataset:
            print(f"Skip {dataset_name}")
            continue

        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, _ = build_dataset(dataset_name, eval=False)

        batch_size = 128
        total_epochs = 40
        if not args.stage1:
            total_epochs = 1
        train(
            dataset=data, 
            model=model, 
            label=label, 
            template=template,
            processor=processor,
            total_epochs=total_epochs,
            dataset_name=dataset_name,
            combdataset=combdataset,
            eval_datasets=eval_datasets,
            batch_size=batch_size,
            collect_only=not args.stage1,
            eval=True,
            save_dir=args.save_dir,
        )
        
    # model.save_pretrained("lora")    
    # Stage2. Replay

    if args.stage2:
        for name, param in model.named_modules():
            if hasattr(param, "loramoe_router"):
                param.close_task()
                param.change_router_state(activate=True)
                
        model.print_trainable_parameters()
        torch.cuda.empty_cache()
        print(f"Training on {len(combdataset.labels)} classes/images")
        train(
            dataset=combdataset, 
            model=model, 
            label=combdataset.labels, 
            template=template,
            processor=processor,
            total_epochs=100,
            eval_datasets=eval_datasets,
            wrapper=False,
            batch_size=64,
            eval=True,
            accumulation_steps=1,
            save_dir=args.save_dir,
            dataset_name="replay",
        )

    run.finish()
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        default="imagenet_a", 
        help="Dataset name or list of dataset names"
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Path to the pretrained model"
    )
    
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Stage 1 training"
    )

    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Stage 2 training"
    )

    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        help="Path to save the model"
    )
    
    parser.add_argument(
        "--train_dataset",
        default="",
        type=str,
        help="Dataset ID for training"
    )


    args = parser.parse_args()
    args.dataset = args.dataset.split(",")

    if args.train_dataset == "":
        args.train_dataset = args.dataset
    if isinstance(args.train_dataset, str):
        args.train_dataset = [args.train_dataset]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print(args)
    return args

if __name__ == "__main__":
    main(get_args())