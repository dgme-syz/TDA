import torch

class ClipImageOutput:
    loss: torch.Tensor
    prob_map: torch.Tensor
    pred: torch.Tensor
    clip_image_logits: torch.Tensor
    image_embeds: torch.Tensor
    
    def __init__(
        self, 
        loss: torch.Tensor,
        prob_map: torch.Tensor,
        pred: torch.Tensor,
        clip_image_logits: torch.Tensor,
        image_embeds: torch.Tensor
    ):
        self.loss = loss
        self.prob_map = prob_map
        self.pred = pred
        self.clip_image_logits = clip_image_logits
        self.image_embeds = image_embeds
        
    def __iter__(self):
        return iter([
            self.loss, 
            self.prob_map, 
            self.pred, 
            self.clip_image_logits, 
            self.image_embeds
        ])