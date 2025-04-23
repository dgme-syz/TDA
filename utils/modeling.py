from transformers import CLIPModel
import torch
from torch import nn

class CLIPCLS(nn.Module):
    def __init__(self, clip: CLIPModel, num_classes: int):
        super().__init__()
        self.clip = clip
        self.fc = nn.Linear(512, num_classes).to(
            device=clip.device, dtype=torch.bfloat16
        )
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, images):
        image_embeds = self.clip.base_model.model.get_image_features(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logits = self.fc(image_embeds)
        return logits