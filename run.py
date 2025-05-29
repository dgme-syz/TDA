import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from utils.tools import eval_all, enable_task, enable_router, load
import os
from peft import PeftModel, PeftConfig, get_peft_model
from utils.lora_moe import LoraMoEConfig
import random
import numpy as np
from safetensors.torch import load_file
import torch.nn.functional as F
from dataclasses import dataclass, field
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 42
data_type = torch.float32

@dataclass
class INFO:
    choose_detail: dict = field(
        default_factory=lambda: [{}],
        metadata={"help": "choose detail"}
    )
    avg_detail: dict = field(
        default_factory=lambda: [],
        metadata={"help": "avg detail"}
    )

    def plot_avg_heatmap(self, resume_dir, num_tasks=11):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
        q_proj_avg = np.concatenate([x["q_proj"] for x in self.avg_detail], axis=0).reshape(-1, num_tasks).astype(np.int32) // 12
        k_proj_avg = np.concatenate([x["k_proj"] for x in self.avg_detail], axis=0).reshape(-1, num_tasks).astype(np.int32) // 12
        v_proj_avg = np.concatenate([x["v_proj"] for x in self.avg_detail], axis=0).reshape(-1, num_tasks).astype(np.int32) // 12
        print(q_proj_avg.shape, k_proj_avg.shape, v_proj_avg.shape)

        q_proj_avg = q_proj_avg / q_proj_avg.sum(axis=1, keepdims=True)
        k_proj_avg = k_proj_avg / k_proj_avg.sum(axis=1, keepdims=True)
        v_proj_avg = v_proj_avg / v_proj_avg.sum(axis=1, keepdims=True)
        sns.heatmap(q_proj_avg, ax=axes[0], annot=True, fmt=".3f", cmap="YlGnBu")
        axes[0].set_title("q_proj_avg")
        sns.heatmap(k_proj_avg, ax=axes[1], annot=True, fmt=".3f", cmap="YlGnBu")
        axes[1].set_title("k_proj_avg")
        sns.heatmap(v_proj_avg, ax=axes[2], annot=True, fmt=".3f", cmap="YlGnBu")
        axes[2].set_title("v_proj_avg")
        plt.tight_layout()
        plt.savefig(os.path.join(resume_dir, "avg_heatmap.png"), dpi=800)
        plt.show()

info = INFO()
def look_hook(module, input, output):
    # to see some details about router
    name = module.name

    if not hasattr(module, "loramoe_router") or "text" in name: 
        return output
    router = module.loramoe_router["default"]
    dtype = router.weight.dtype
    input = input[0]
    x = module._cast_input_dtype(input, dtype)
    B, S, H = x.shape
    hidden_states = x[:, 0, :] # use [CLS] token
    weight = F.normalize(router.weight, dim=-1)
    sim = F.normalize(hidden_states, dim=-1) @ weight.T
    num_tasks = sim.shape[-1]
    choose = sim.argmax(dim=-1).cpu().numpy()
    counts = np.bincount(choose, minlength=num_tasks)
    if info.choose_detail[-1].get(name) is None:
        info.choose_detail[-1][name] = counts
    else:
        info.choose_detail[-1][name] += counts
    return output

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
        torch_dtype=data_type,
        attn_implementation="flash_attention_2" if data_type in [torch.float16, torch.bfloat16] else None
    ).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    model = load(base_model=model, resume_dir=args.resume, dtype=data_type)
    model.text_mode = {"default": -1}
    for name, module in model.named_modules():
        if hasattr(module, "loramoe_router"):
            module.register_forward_hook(look_hook)
            module.name = name
            if "text" in name:
                module.text_mode = model.text_mode

    # run = wandb.init(project="CLIP-MOE-EVAL", config=None, name="exp")
    enable_task(model, None)
    # Eval
    eval_all(
        model=model, 
        processor=processor, 
        datasets=args.dataset, 
        wandb_use=False,
        info=info,
    )
    info.plot_avg_heatmap(resume_dir=args.resume, num_tasks=11)
    # run.finish()
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        default="imagenet_a", 
        help="Dataset name or list of dataset names"
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