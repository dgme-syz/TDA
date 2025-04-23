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
import re
import numpy as np
from .classes import DataWrapper
import torch.nn.functional as F
import torch.nn as nn
from data import build_dataset
import os
import wandb


wordnet_pattern = re.compile(r"n[0-9]{8}")

def convert_wordnet_to_cls(name: str):
    from nltk.corpus import wordnet
    return wordnet.synset_from_pos_and_offset(
        'n', int(re.search(r"^n(\d{8})$", name).group(1))
    ).name().split('.')[0]

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def extract_clip_text_weights(
    class_label: list[str],
    template: Union[List[str], str],
    clip_model: CLIPModel, 
    processor: CLIPProcessor, 
    is_train: bool = False,
    **kwargs
) -> List[torch.Tensor]:
    # is not is_train use no_grad, otherwise not use no_grad
    manager = torch.no_grad() if not is_train else torch.enable_grad()
    with manager:
        clip_weights = []
        
        for cls in class_label:
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
    use_cache: bool, 
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
        shuffle=True, 
        pin_memory=torch.cuda.is_available(), 
        num_workers=8 # -> 2 进程 4 线程
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
            
            if use_cache:
                cache.update_pos_cache(pred, image_embeds, loss)
                cache.update_neg_cache(pred, image_embeds, loss, True, prob_map, prop_entropy)
            
            final_logits: torch.Tensor = clip_image_logits.clone()

            if use_cache:
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

def eval_all(
    model, 
    processor, 
    datasets, 
    template,
    cache_config="./configs",
    use_cache=False,
):
    model.eval()
    for dataset_name in datasets:
        torch.cuda.empty_cache()
        cache = CacheModule.load_from_yaml(os.path.join(cache_config, dataset_name + ".yaml"))   
        print(
            f"Loading dataset: {dataset_name}"
        )
        
        data, label, _ = build_dataset(dataset_name)
        clip_weights = extract_clip_text_weights(
            class_label=label, template=template, clip_model=model, processor=processor
        )
            
        results = eval(
            dataset=data, 
            model=model, 
            cache=cache, 
            label=label, 
            text_embeds=clip_weights, 
            use_cache=use_cache
        )
        
        print(
            f"Results for {dataset_name}[ACC]: {results}"
        )      
        wandb.log({f"{dataset_name}_test_acc": results}, commit=True)

def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: torch.Tensor | None = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """

    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1, dtype=torch.float)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

from torch.amp import autocast, GradScaler

def train(
    dataset: Dataset,
    model: CLIPModel,
    label: list[str],
    template: Union[List[str], str],
    processor: CLIPProcessor,
    eval_datasets: List[str],
    save_dir: str,
    dataset_name: str | None = None,
    total_epochs: int = 1000,
    wrapper: bool = True,
    combdataset=None,
    batch_size: int = 256,
    eval: bool = False,
    accumulation_steps: int = 4,
    collect_only: bool = False,
    **kwargs
):
    if dataset_name is not None and dataset_name != "replay":
        eval_datasets.append(dataset_name)
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
        lr=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=1e-6)

    scaler = GradScaler(enabled=False)  
    amp_dtype = torch.bfloat16 

    for epoch in range(total_epochs):
        model.train()
        res = { "correct": 0, "total": 0, "loss": 0 }
        pbar = tqdm(data_loader, desc="Processed train images: ")

        optimizer.zero_grad()
        accumulation_counter = 0

        for x in pbar:
            images, target = x["image"], x["label"]
            if combdataset is not None:
                for i in range(target.size(0)):
                    lab = label[target[i].item()]
                    if lab not in combdataset.labels:
                        combdataset.add(images[0][i].cpu().detach(), lab)
            if collect_only: continue

            if not isinstance(images, (tuple, list)):
                images = (images,)
            images = torch.cat(images, dim=0).to(model.device)
            target = target.view(-1).to(model.device)

            with autocast(dtype=amp_dtype, device_type="cuda"):
                image_embeds = model.get_image_features(images)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                text_embeds = extract_clip_text_weights(
                    class_label=label, template=template, clip_model=model, processor=processor, is_train=True
                )
                target_embeds = text_embeds[:, target]
                logits_per_text = 100. * image_embeds @ target_embeds
                loss = clip_loss(logits_per_text)
            for _, param in model.named_parameters():
                if hasattr(param, "router_loss"):
                    loss = loss + param.router_loss

            if not torch.isfinite(loss):
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

            with torch.no_grad():
                clip_logits = 100. * image_embeds @ text_embeds
                union_pred = clip_logits.argmax(dim=-1)
                res["correct"] += torch.sum(union_pred == target).item()
                res["total"] += len(target)
                res["loss"] += loss.item() * len(target)

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

        if not collect_only:
            scheduler.step()
            print(f"Epoch {epoch}: Loss: {res['loss'] / res['total']:.4f}, Accuracy: {res['correct'] / res['total']:.4f}")
            wandb.log(
                {
                    f"{dataset_name}_train_loss": res["loss"] / res["total"], f"{dataset_name}_train_acc": res["correct"] / res["total"]
                }, 
                commit=True
            )
            acc = res["correct"] / res["total"]
            if (1 - acc) < 0.008:
                print(f"Early stopping at epoch {epoch} with accuracy {acc:.4f}")
                break
            if (epoch + 1) % 1 == 0:
                if eval:
                    if dataset_name == "replay":
                        eval_datasets = ["fgvc", "dtd"]
                    eval_all(
                        model=model,
                        processor=processor,
                        datasets=eval_datasets,
                        use_cache=False,
                        template=template,
                    )
                model.save_pretrained(os.path.join(save_dir, f"{dataset_name}_{batch_size}_{total_epochs}"))
