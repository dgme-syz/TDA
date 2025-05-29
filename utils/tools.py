from transformers import CLIPModel, CLIPProcessor
from typing import List, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from .classes import ClipImageOutput
import re
import numpy as np
from .classes import DataWrapper
import torch.nn as nn
from data import build_dataset
import os
import wandb
from torch.amp import autocast, GradScaler
from peft import PeftModel, get_peft_model
from peft.utils import id_tensor_storage
import collections
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file
from .lora_moe import LoraMoEConfig
import json
from typing import Callable
import gc


wordnet_pattern = re.compile(r"n[0-9]{8}")

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def extract_clip_text_weights(
    class_label: list[str],
    template: Callable[[str], str],
    clip_model: CLIPModel, 
    processor: CLIPProcessor, 
    is_train: bool = False,
    **kwargs
) -> List[torch.Tensor]:
    # is not is_train use no_grad, otherwise not use no_grad
    with torch.no_grad() if not is_train else torch.enable_grad():
        clip_weights = []
        for cls in class_label:
            prompts = template(cls)
            input_ids = processor(
                text=prompts, return_tensors="pt", padding=True
            )["input_ids"].to(clip_model.device)
            u = clip_model.class_dict[cls]
            for name, module in clip_model.named_modules():
                if hasattr(module, "loramoe_router") and "text" in name:
                    module.text_mode["default"] = u
            # print(input_ids.shape, input_ids)
            cls_embed = clip_model.get_text_features(input_ids) # batch_size x hidden_size
            cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
            clip_weights.append(cls_embed.squeeze(0)) 
        
        clip_weights = torch.stack(clip_weights, dim=1)  
        assert len(clip_weights.shape) == 2
        return clip_weights # hidden_size x num_classes

def get_clip_text_weights(
    targets: Union[List[str], str],
    template: Callable[[str], str],
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    class_label: list[str],
    require_grad: bool = False
) -> torch.Tensor:
    with torch.enable_grad() if require_grad else torch.no_grad():
        prompts = []
        targets = targets.tolist()
        task_id = clip_model.class_dict[class_label[targets[0]]]
        for cls in targets:
            prompts.append(template(class_label[cls]))
            # print(prompts[-1])
        input_ids = processor(
            text=prompts, return_tensors="pt", padding=True
        )["input_ids"].to(clip_model.device)

        # switch to the task_id
        for name, module in clip_model.named_modules():
            if hasattr(module, "loramoe_router") and "text" in name:
                module.text_mode["default"] = task_id
        
        # get text features
        cls_embed = clip_model.get_text_features(input_ids) # batch_size x hidden_size
        cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
            
    return cls_embed # batch_size x hidden_size

        
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
    label: list[str],
    **kwargs
):
    model.eval()
    num_classes = len(label)
    acc = {
        "correct": 0, "total": 0
    }
    
    data_loader = DataLoader(
        dataset=DataWrapper(dataset, augmix=False), 
        batch_size=1, 
        shuffle=False, 
        pin_memory=torch.cuda.is_available(), 
        num_workers=8, # -> 2 进程 4 线程
        drop_last=False
    )
    
    pbar = tqdm(data_loader, desc="Processed test images: ") 
    # x1 = [False for _ in range(len(data_loader))]
    with torch.no_grad():
        for x in pbar:
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
            
            final_logits: torch.Tensor = clip_image_logits.clone()
            
            union_pred = final_logits.argmax(dim=-1)
            acc["correct"] += torch.sum(union_pred == target).item()
            acc["total"] += len(target)
            accuracy = acc["correct"] / acc["total"] * 100
            pbar.set_postfix({
                "acc": f"{accuracy:.2f}%" 
            })

    return acc["correct"] / acc["total"]

def eval_all(
    model, 
    processor, 
    datasets, 
    wandb_use=True,
    info=None,
):
    model.eval()
    for dataset_name in datasets:
        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, template = build_dataset(dataset_name)
        clip_weights = extract_clip_text_weights(
            class_label=label, template=template, clip_model=model, processor=processor
        )
            
        results = eval(
            dataset=data, 
            model=model, 
            label=label, 
            text_embeds=clip_weights, 
        )
        
        print(
            f"Results for {dataset_name}[ACC]: {results}"
        )      
        if wandb_use:
            wandb.log({f"{dataset_name}_test_acc": results}, commit=True)
        if info is not None:
            # print(f"Router choose detail: {info.choose_detail}")
            # save info.choose_detail to wandb saves
            counts = {
                "q_proj": np.zeros(11, dtype=np.int32), 
                "k_proj": np.zeros(11, dtype=np.int32), 
                "v_proj": np.zeros(11, dtype=np.int32),
            }
            info.avg_detail.append(counts)
            for k, v in info.choose_detail[-1].items():
                for key, _ in info.avg_detail[-1].items():
                    if key in k:
                        info.avg_detail[-1][key] += v

            info.choose_detail.append({})



def enable_task(model, task_id: int | None = None):
    for name, param in model.named_modules():
            if hasattr(param, "loramoe_router"):
                param.enable_task(task_id)
                param.change_router_state(activate=False)
def enable_router(model):
    for name, param in model.named_modules():
            if hasattr(param, "loramoe_router"):
                param.close_task()
                param.change_router_state(activate=True)
def zero_router(model, task_id):
    for name, param in model.named_modules():
            if hasattr(param, "loramoe_router"):
                param.zero_router_init(task_id)

def train(
    dataset: Dataset,
    model: CLIPModel,
    label: list[str],
    template: Callable[[str], str],
    processor: CLIPProcessor,
    eval_datasets: List[str],
    save_dir: str,
    dataset_name: str | None = None,
    wrapper: bool = True,
    batch_size: int = 256,
    eval: bool = False,
    accumulation_steps: int = 1,
    task_id: int | None = None,
    data_type: torch.dtype = torch.bfloat16,
    **kwargs
):
    if task_id is not None:
        for i in range(task_id, 11):
            zero_router(model, i)
    if wrapper:
        dataset = DataWrapper(dataset, augmix=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=8,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=1e-3,
    )
    total_epochs = max(25, (2500 + len(data_loader) - 1) // len(data_loader))
    print(f"Total epochs: {total_epochs}, batch size: {batch_size}, accumulation steps: {accumulation_steps}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=1e-4)

    scaler = GradScaler(enabled=False)  
    amp_dtype = data_type

    for epoch in range(total_epochs):
        model.train()
        if task_id is not None:
            for name, param in model.named_modules():
                if hasattr(param, "loramoe_router"):
                    param.enable_task(task_id)
                    param.sample = 2000
        

        res = { "correct": 0, "total": 0, "loss": 0 }
        pbar = tqdm(data_loader, desc="Processed train images: ")
        optimizer.zero_grad()
        accumulation_counter = 0

        for x in pbar:
            images, target = x["image"], x["label"]

            if not isinstance(images, (tuple, list)):
                images = (images,)
            images = torch.cat(images, dim=0).to(model.device)
            target = target.view(-1).to(model.device)

            with autocast(dtype=amp_dtype, device_type="cuda"):
                image_embeds = model.get_image_features(images)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                text_embeds = get_clip_text_weights(
                    targets=target, 
                    template=template, 
                    clip_model=model, 
                    processor=processor, 
                    class_label=label, 
                    require_grad=True
                )
                # text_weights = extract_clip_text_weights(
                #     class_label=label, 
                #     template=template, 
                #     clip_model=model, 
                #     processor=processor, 
                #     is_train=True
                # )
                # text_embeds = text_weights[:, target]
                logits_per_text = 100. * image_embeds @ text_embeds.T
                # logits_per_text = 100. * image_embeds @ text_embeds
                loss = clip_loss(logits_per_text)

            if not torch.isfinite(loss):
                # For some reason, the loss is not finite, skip this batch
                # clean 
                # del loss, image_embeds, text_embeds, logits_per_text, text_weights
                del loss, image_embeds, text_embeds, logits_per_text
                torch.cuda.empty_cache()
                gc.collect()
                continue

            loss.backward()
            accumulation_counter += 1
            if accumulation_counter == accumulation_steps:
                scaler.unscale_(optimizer)
                grads_finite = all(
                    p.grad is None or torch.all(torch.isfinite(p.grad)) for p in model.parameters()
                )
                if grads_finite:
                    optimizer.step()
                    scaler.update()

                optimizer.zero_grad()
                accumulation_counter = 0

            with torch.no_grad() and autocast(dtype=amp_dtype, device_type="cuda"):
                total_classes = torch.arange(len(label), device='cpu')
                text_weights = get_clip_text_weights(
                    targets=total_classes, 
                    template=template, 
                    clip_model=model, 
                    processor=processor, 
                    class_label=label, 
                    require_grad=False
                )
                clip_logits = 100. * image_embeds.detach() @ text_weights.T
                del text_weights
                # clip_logits = 100. * image_embeds.detach() @ text_weights.detach()
                # del image_embeds, text_embeds, logits_per_text, text_weights
                union_pred = clip_logits.argmax(dim=-1)
                res["correct"] += torch.sum(union_pred == target).item()
                res["total"] += len(target)
                res["loss"] += loss.item() * len(target)
            del loss
            pbar.set_postfix({
                f"{dataset_name}_loss": f"{res['loss'] / res['total']:.4f}",
                "acc": f"{res['correct'] / res['total']:.2f}"
            })
        if accumulation_counter > 0:
            scaler.unscale_(optimizer)
            grads_finite = all(
                p.grad is None or torch.all(torch.isfinite(p.grad)) for p in model.parameters()
            )
            if grads_finite:
                optimizer.step()
                scaler.update()

            optimizer.zero_grad()
        scheduler.step()
        print(f"Epoch {epoch}: Loss: {res['loss'] / res['total']:.4f}, Accuracy: {res['correct'] / res['total']:.4f}")
        wandb.log(
            {
                f"{dataset_name}_train_loss": res["loss"] / res["total"], f"{dataset_name}_train_acc": res["correct"] / res["total"]
            }, 
            commit=True
        )
        acc = res["correct"] / res["total"]

        if (
            (epoch + 1) % 10 == 0 
            or (1 - acc) < 0.005
            or epoch + 1 == total_epochs
        ):
            if eval:
                enable_task(model, None)
                x = [dataset_name, ]
                # if epoch + 1 == total_epochs or (1 - acc) < 0.005:
                x = eval_datasets
                eval_all(
                    model=model,
                    processor=processor,
                    datasets=x,
                )
                if task_id is not None:
                    enable_task(model, task_id)
            if (1 - acc) < 0.005:
                save(model, save_dir, -1, True, "router")
                save(model, save_dir, task_id, True, dataset_name)
                print(f"Early stopping at epoch {epoch} with accuracy {acc:.4f}")
                break
        save(model, save_dir, task_id, True, dataset_name)
        save(model, save_dir, -1, True, "router")

def save(model: PeftModel, save_dir: str, task_id: int | None, safe_serialization: bool = True, prefix: str | None = None):
    if task_id is None:
        # save config
        model.peft_config["default"].save_pretrained(save_dir)
    else:
        state_dict = model.state_dict()
        if task_id == -1:
            # save router
            to_returns = {k: state_dict[k] for k in state_dict if "loramoe_router" in k}
        else:
            to_returns = {
                k: state_dict[k] for k in state_dict 
                if any(f"loramoe_{m}.default.{task_id}" in k for m in ["A", "B"])
            }
        if safe_serialization:
            ptrs = collections.defaultdict(list)
            for name, tensor in to_returns.items():
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    ptrs[id(tensor)].append(name)
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    to_returns[shared_tensor_name] = to_returns[shared_tensor_name].clone()
        safe_save_file(
            to_returns,
            os.path.join(save_dir, f"{prefix}.safetensors"),
            metadata={"format": "pt"}
        )
    with open(os.path.join(save_dir, "class_dict"), "w") as f:
        json.dump(model.class_dict, f)

def load(base_model: nn.Module, resume_dir: str, dtype) -> PeftModel:
    model = get_peft_model(base_model, LoraMoEConfig.from_pretrained(resume_dir)).to(dtype)

    class_dict = {}
    for x in os.listdir(resume_dir):
        if x.endswith(".safetensors"):
            print(f"Loading {x}")
            state_dict = load_file(os.path.join(resume_dir, x))
            print("unexpected keys: ", model.load_state_dict(state_dict, strict=False)[1])
        elif x == "class_dict":
            with open(os.path.join(resume_dir, x), "r") as f:
                class_dict = json.load(f)
    model.class_dict = class_dict
    return model.to(dtype)