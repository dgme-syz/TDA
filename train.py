import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from data import build_dataset
from utils.tools import train, enable_task, save, load
from data import dataset_list
from peft import get_peft_model
from utils.lora_moe import LoraMoEConfig

import os
import wandb
import random
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# python3 train.py --dataset fgvc,caltech101,dtd,eurosat,oxford_flowers,oxford_pets
# python3 train.py --dataset fgvc,caltech101,cifar100,dtd,eurosat,oxford_flowers,food101,mnist,oxford_pets,stanford_cars,sun397
seed = 42
data_type=torch.float32
def main(args):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    torch.manual_seed(seed)
    # torch.autograd.set_detect_anomaly(True)
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", 
        device_map="auto", 
        torch_dtype=data_type,
        attn_implementation="flash_attention_2" if data_type in [torch.float16, torch.bfloat16] else None
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # LoRA
    
    if args.resume != "":
        model = load(base_model=model, resume_dir=args.resume, dtype=data_type)
    else:
        target = []
        for name, _ in model.named_modules():
            if any([x in name for x in ["q_proj", "v_proj", "k_proj"]]) and (
                # ("text" not in name) or
                # any([x in name for x in [".8.", ".9.", ".10.", ".11."]])
                True
            ):
                target.append(name)
        config = LoraMoEConfig(
            r=32,
            top_k=1, 
            num_tasks=len(args.dataset),
            lora_alpha=16,
            target_modules=target,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights="pissa"
        )
        model = get_peft_model(model, config).to(data_type)
        model.class_dict = {}
    print(model)
    # print(config)
    run = wandb.init(project="CLIP-MOE", config=None, name="exp")

    eval_datasets = []
    
    # save config
    model.text_mode = {"default": -1}
    for name, module in model.named_modules():
        if hasattr(module, "loramoe_router"):
            module.name = name
            if "text" in name:
                module.text_mode = model.text_mode
        
    save(model, save_dir=args.save_dir, task_id=None)
    # Stage1. Train LoRA
    for i in range(len(args.dataset)):
        enable_task(model, i)
        model.print_trainable_parameters()
        dataset_name = args.dataset[i]
        if dataset_name is not None and dataset_name != "replay":
            eval_datasets.append(dataset_name)

        if dataset_name not in args.train_dataset:
            print(f"Skip {dataset_name}")
            continue
        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, temp = build_dataset(dataset_name, eval=False)
        for x in label:
            model.class_dict[x] = i
        batch_size = 128
        train(
            dataset=data, 
            model=model, 
            label=label, 
            template=temp,
            processor=processor,
            dataset_name=dataset_name,
            eval_datasets=eval_datasets,
            batch_size=batch_size,
            eval=True,
            save_dir=args.save_dir,
            task_id=i,
            data_type=data_type,
        )
        # save router only
        save(model, args.save_dir, -1, True, "router")
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
    for x in args.dataset:
        if x not in dataset_list:
            raise ValueError(f"Dataset {x} not found in dataset list.")
    if args.train_dataset == "":
        args.train_dataset = args.dataset
    if isinstance(args.train_dataset, str):
        args.train_dataset = args.train_dataset.split(",")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print(args)
    return args

if __name__ == "__main__":
    main(get_args())