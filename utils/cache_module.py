from queue import PriorityQueue
import yaml
import torch
from typing import List, Union
from collections import OrderedDict

class CacheModule:
    pos_enabled: bool
    neg_enabled: bool
    pos_params: OrderedDict
    neg_params: OrderedDict
    pos_cache: OrderedDict
    neg_cache: OrderedDict
    
    def __init__(
        self, 
        pos_params: OrderedDict,
        neg_params: OrderedDict,
        pos_enabled: bool,
        neg_enabled: bool,
        **kwargs
    ):
        self.pos_params = pos_params
        self.neg_params = neg_params
        self.pos_enabled = pos_enabled
        self.neg_enabled = neg_enabled
        self.pos_cache = OrderedDict()
        self.neg_cache = OrderedDict()
        self.tick = 0
    
    def __str__(self):
        return f"CacheModule(pos_enabled={self.pos_enabled}, neg_enabled={self.neg_enabled})\n" + \
            f"Positive Cache Params: {self.pos_params}\n" + \
            f"Negative Cache Params: {self.neg_params}"

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        config = yaml.safe_load(open(yaml_path))
        return cls(
            pos_params=config['positive'],
            neg_params=config['negative'],
            pos_enabled=config['positive']['enabled'],
            neg_enabled=config['negative']['enabled']
        )

    @torch.no_grad()
    def update_cache(
        self, 
        cache: OrderedDict,
        pred: torch.Tensor, 
        image_features: torch.Tensor,
        loss: torch.Tensor,
        include_prob_map=False,
        prob_map: torch.Tensor=None,
        cache_params: dict=None,
        **kwargs
    ): 
        r"""
            Update cache with new features and loss, maintaining the maximum shot capacity.
        """
        with torch.no_grad():
            batch_size = pred.shape[0]
            for i in range(batch_size):
                pred = pred[i].item()
                if pred not in cache:
                    cache[pred] = PriorityQueue(maxsize=cache_params["shot_capacity"] + 1)
                # save less loss
                item = (-loss[i].item(), self.tick, image_features[i]) if not include_prob_map else (-loss[i].item(), self.tick, image_features[i], prob_map[i])
                self.tick += 1
                cache[pred].put(item)
                if cache[pred].full():
                    # pop the biggest loss
                    cache[pred].get() # ensure cache size is always less than or equal to shot_capacity
        
    def update_pos_cache(
        self, 
        pred: torch.Tensor,
        image_features: torch.Tensor,
        loss: torch.Tensor,
    ):
        if self.pos_enabled:
            self.update_cache(self.pos_cache, pred, image_features, loss, cache_params=self.pos_params)
        
    def update_neg_cache(
        self,
        pred: torch.Tensor, 
        image_features: torch.Tensor,
        loss: torch.Tensor,
        include_prob_map: bool,
        prob_map: torch.Tensor,    
        prop_entropy: float,
    ):
        if self.neg_enabled:
            lo = self.neg_params['entropy_threshold']['lower']
            hi = self.neg_params['entropy_threshold']['upper']
            
            if prop_entropy > lo and prop_entropy < hi:
                self.update_cache(self.neg_cache, pred, image_features, loss, include_prob_map, prob_map, cache_params=self.neg_params)
    
    def _compute_extra_logits_with_cache(
        self, 
        image_features: torch.Tensor,
        cache: OrderedDict,
        cache_params: dict,
        num_classes: int=None,
        mask_thresholds: List[float]=None
    ) -> torch.Tensor:
        r"""
            Compute similarity logits with cache, for all the samples in the cache
        """
        with torch.no_grad():
            cache_keys, cache_values = [], []
            for cls_idx in sorted(cache.keys()):
                for item in cache[cls_idx].queue:
                    # extract image features
                    cache_keys.append(item[2]) # see update_cache, 1 x embedding_size
                    if mask_thresholds:
                        # prob_map
                        # print(item[2].shape)
                        cache_values.append(item[3]) # 1 x num_classes,
                    else:
                        cache_values.append(cls_idx)
            
            data_dtype = image_features.dtype
            cache_keys = torch.stack(cache_keys, dim=0).to(image_features.device) # sample_size x embedding_size
            assert len(cache_keys.shape) == 2, "Cache keys shape mismatch"
            if mask_thresholds:
                cache_values = torch.stack(cache_values, dim=0) # sample_size x num_classes
                cache_values = ((
                    (cache_values > mask_thresholds[0]) & 
                    (cache_values < mask_thresholds[1])
                ).type(torch.int8)).to(image_features.device).to(data_dtype) # sample_size x num_classes
            else:
                cache_values = torch.nn.functional.one_hot(
                    torch.tensor(cache_values).to(torch.int64), num_classes
                ).to(image_features.device).to(data_dtype) # sample_size x num_classes
                assert len(cache_values.shape) == 2, "Cache values shape mismatch"
            
            affinity = image_features @ cache_keys.T # batch_size x sample_size
            
            alpha, beta = cache_params['alpha'], cache_params['beta']
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            
            return alpha * cache_logits
        
    def compute_pos_logits(
        self, 
        image_features: torch.Tensor,
        num_classes: int,
        **kwargs
    ) -> Union[torch.Tensor, None]:
        if self.pos_enabled and len(self.pos_cache) > 0:
            return self._compute_extra_logits_with_cache(
                image_features, self.pos_cache, self.pos_params, num_classes, mask_thresholds=None
            )
        else:
            return None
        
    def compute_neg_logits(
        self,
        image_features: torch.Tensor,
        **kwargs
    ) -> Union[torch.Tensor, None]:
        if self.neg_enabled and len(self.neg_cache) > 0:
            lo = self.neg_params['mask_threshold']['lower']
            hi = self.neg_params['mask_threshold']['upper']
            return self._compute_extra_logits_with_cache(
                image_features, self.neg_cache, self.neg_params, mask_thresholds=(lo, hi)
            )
        else:
            return None    