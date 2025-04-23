# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose

from .config import LoraMoEConfig
from .dora import DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer, _DoraConvNdLayer
from torch.utils.checkpoint import checkpoint

class LoraMoELayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("loramoe_A", "loramoe_B", "loramoe_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "num_tasks")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.num_tasks = {}
        self.top_k = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.task_id = {}
        self.lora_dropout = nn.ModuleDict({})
        self.loramoe_A = nn.ModuleDict({})
        self.loramoe_B = nn.ModuleDict({})
        self.router_loss = 0
        self.router_enable = {}
        # soft weight
        self.loramoe_router = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        num_tasks,
        top_k, 
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.top_k[adapter_name] = top_k
        self.num_tasks[adapter_name] = num_tasks
        self.lora_alpha[adapter_name] = lora_alpha
        self.task_id[adapter_name] = None
        self.router_enable[adapter_name] = False

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.loramoe_router[adapter_name] = nn.Linear(self.in_features, num_tasks) # task router/weight
        nn.init.kaiming_uniform_(self.loramoe_router[adapter_name].weight, a=math.sqrt(5))
        self.loramoe_A[adapter_name] = nn.ModuleList([nn.Linear(self.in_features, r, bias=False) for _ in range(num_tasks)])
        self.loramoe_B[adapter_name] = nn.ModuleList([nn.Linear(r, self.out_features, bias=lora_bias) for _ in range(num_tasks)])
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            for i in range(num_tasks):
                nn.init.zeros_(self.loramoe_B[adapter_name][i].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        num_tasks = self.num_tasks[adapter_name]
        # default initialization for router
        
        
        if adapter_name in self.loramoe_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                for i in range(num_tasks):
                    nn.init.kaiming_uniform_(self.loramoe_A[adapter_name][i].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                for i in range(num_tasks):
                    nn.init.normal_(self.loramoe_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            for i in range(num_tasks):
                nn.init.zeros_(self.loramoe_B[adapter_name][i].weight)
            if self.lora_bias[adapter_name]:
                for i in range(num_tasks):
                    nn.init.zeros_(self.loramoe_B[adapter_name][i].bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype

        if bnb_param_type:
            # check without importing bitsandbytes and robust to bnb_4bit_quant_storage=float*
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        num_tasks = self.num_tasks[adapter_name]
        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        for i in range(num_tasks):
            self.loramoe_A[adapter_name][i].weight.data = Rr.contiguous().clone()
            self.loramoe_B[adapter_name][i].weight.data = Qr.contiguous().clone()

        weight_tensor.data -= scale_factor * self.loramoe_B[adapter_name][0].weight @ self.loramoe_A[adapter_name][0].weight
        if bnb_param_type == "4bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                quant_type=orig_weight.quant_type,
                quant_storage=orig_weight.quant_storage,
                compress_statistics=orig_weight.compress_statistics,
                module=orig_weight.module,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        elif bnb_param_type == "8bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                requires_grad=orig_weight.requires_grad,
                has_fp16_weights=orig_weight.has_fp16_weights,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor.to(dtype)
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        num_tasks = self.num_tasks[adapter_name]

        for i in range(num_tasks):
            self.loramoe_A[adapter_name][i].weight.data = lora_A.clone()
            self.loramoe_B[adapter_name][i].weight.data = lora_B.clone()
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def corda_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize CorDA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)

        # Calculate WC from covariance matrix
        if not hasattr(linear, "eigens"):
            raise ValueError(
                "`eigens` attribute not found for layer, please run `preprocess_corda` first. "
                "More information can be found at examples/corda_finetuning/README.md."
            )
        eigens = linear.eigens
        U = eigens.U_WC
        S = eigens.S_WC
        V = eigens.V_WC
        r = self.r[adapter_name]

        # nan or inf check
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise ValueError(
                "Invalid value found in matrix S. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise ValueError(
                "Invalid value found in matrix U. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise ValueError(
                "Invalid value found in matrix V. Please file an issue at https://github.com/huggingface/peft/issues."
            )

        # Sanity check
        if U.size(0) != out_dim or U.size(1) != r:
            raise ValueError(
                f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )
        if S.size(0) != r:
            raise ValueError(
                f"Matrix S size mismatch: {S.size()} vs. ({r},). Please make sure the `lora_config` and `model` argument "
                "of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, "
                "please make sure the cache is built with the same model and LoRA rank."
            )
        if V.size(0) != in_dim or V.size(1) != r:
            raise ValueError(
                f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )

        # Apply alpha
        S /= self.scaling[adapter_name]

        # Init lora_A and lora_B weights
        num_tasks = self.num_tasks[adapter_name]
        lora_A = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        lora_B = U.mul(S.sqrt()).contiguous()

        for i in range(num_tasks):
            self.loramoe_A[adapter_name][i].weight.data = lora_A.clone()
            self.loramoe_B[adapter_name][i].weight.data = lora_B.clone()
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

        # Remove redundant fields
        del linear.eigens

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        num_tasks = self.num_tasks[adapter_name]
        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.loramoe_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(num_tasks):
                self.loramoe_A[adapter_name][i].weight.data = lora_A.clone()
                self.loramoe_B[adapter_name][i].weight.data = lora_B.clone()
        
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        self.lora_magnitude_vector[adapter_name] = nn.ModuleList([])

        for i in range(self.num_tasks[adapter_name]):
            dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))

            lora_A = self.loramoe_A[adapter_name][i].weight
            lora_B = self.loramoe_B[adapter_name][i].weight
            place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
            if self.ephemeral_gpu_offload:
                if lora_A.device.type in ["cuda", "xpu"]:
                    lora_B = lora_B.to(lora_A.device)
                else:
                    if lora_B.device.type not in ["cuda", "xpu"]:
                        if is_xpu_available():
                            lora_B = lora_B.to("xpu")
                        else:
                            lora_B = lora_B.to("cuda")
                    lora_A = lora_A.to(lora_B.device)
            scaling = self.scaling[adapter_name]
            dora_layer.update_layer(
                base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
            )
            self.lora_magnitude_vector[adapter_name].append(dora_layer)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.loramoe_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.loramoe_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = list(set(adapter_names))
        sub_batch_indices_list = [
            [idx for idx, name in enumerate(adapter_names) if name == adapter]
            for adapter in unique_adapters
        ]

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__" or active_adapter not in self.loramoe_A:
                continue

            top_k = self.top_k[active_adapter]
            lora_A = self.loramoe_A[active_adapter]
            lora_B = self.loramoe_B[active_adapter]
            router = self.loramoe_router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            num_tasks = self.num_tasks[active_adapter]
            task_id = self.task_id[active_adapter]
            router_enable = self.router_enable[active_adapter]
            # Prepare sub-batch
            sub_indices = sub_batch_indices_list[i]
            sub_batch = x[sub_indices]
            sub_batch = self._cast_input_dtype(sub_batch, lora_A[0].weight.dtype)
            batch_size, seq_len, hidden_dim = sub_batch.shape

            sub_batch = dropout(sub_batch)
            hidden_states = sub_batch[:, 0, :]

            # Routing
            # if router can be trained
            if router_enable:
                router_logits = router(hidden_states)
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                del router_logits
            elif task_id is not None:
                routing_weights = F.one_hot(
                    torch.tensor([task_id] * (batch_size), device=x.device),
                    num_classes=num_tasks,
                ).float()
                top_k = 1
            else:
                raise NotImplementedError

            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros_like(sub_batch)

            expert_mask = F.one_hot(selected_experts, num_classes=num_tasks).permute(2, 1, 0)

            torch.cuda.empty_cache()  

            for j in range(num_tasks):
                if not expert_mask[j].any():
                    continue
                idx, top_x = torch.where(expert_mask[j])

                current_state = x.index_select(0, top_x) # mini_batch x sequence_length x hidden_dim
                def lora_fn(state):
                    return lora_B[i](lora_A[i](state))

                out = lora_fn(current_state) * (routing_weights.index_select(0, top_x).gather(1, idx[:, None]))[:, None, :]
                final_hidden_states.index_add_(0, top_x, out)

            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
            result[sub_indices] += final_hidden_states * scaling
            result = result.to(torch_result_dtype)

        return result

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)

    def set_task_id(self, task_id: int | None) -> None:
        for active_adapter in self.active_adapters:
            self.task_id[active_adapter] = task_id

    def enable_task(self, task_id: int) -> None:
        for active_adapter in self.active_adapters:
            num_tasks = self.num_tasks[active_adapter]
            for i in range(num_tasks):
                if i != task_id:
                    self.loramoe_A[active_adapter][i].weight.requires_grad = False
                    self.loramoe_B[active_adapter][i].weight.requires_grad = False
                else:
                    self.task_id[active_adapter] = task_id
                    self.loramoe_A[active_adapter][i].weight.requires_grad = True
                    self.loramoe_B[active_adapter][i].weight.requires_grad = True
    def close_task(self) -> None:
        for active_adapter in self.active_adapters:
            num_tasks = self.num_tasks[active_adapter]
            for i in range(num_tasks):
                self.loramoe_A[active_adapter][i].weight.requires_grad = False
                self.loramoe_B[active_adapter][i].weight.requires_grad = False
        self.task_id[active_adapter] = None

    def change_router_state(self, activate: bool) -> None:
        for active_adapter in self.active_adapters:
            if activate:
                self.loramoe_router[active_adapter].weight.requires_grad = True
                self.router_enable[active_adapter] = True
            else:
                self.loramoe_router[active_adapter].weight.requires_grad = False
                self.router_enable[active_adapter] = False

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

def print_current_memory_usage(stage=""):
    allocated_memory = torch.cuda.memory_allocated("cuda") / (1024 ** 2)
    print(f"Memory Usage at {stage}: {allocated_memory:.2f} MB")

class Linear(nn.Module, LoraMoELayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        top_k: int = 1,
        num_tasks: int = 2,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraMoELayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            top_k=top_k,
            num_tasks=num_tasks,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        raise NotImplementedError("This method cannot be merged")
        

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise NotImplementedError("This method cannot be merged")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        raise NotImplementedError("This method cannot be merged")


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        lora_A_keys = self.loramoe_A.keys()

        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.loramoe_A[active_adapter]
            lora_B = self.loramoe_B[active_adapter]
            router = self.loramoe_router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            num_tasks = self.num_tasks[active_adapter]
            task_id = self.task_id[active_adapter]
            router_enable = self.router_enable[active_adapter]
            top_k = self.top_k[active_adapter]
            dtype = lora_A[0].weight.dtype
            x = self._cast_input_dtype(x, dtype)
            # print_current_memory_usage("init")
            B, S, H = x.shape
            x = dropout(x)  
            hidden_states = x[:, 0, :] # use [CLS] token

            end = num_tasks if task_id is None else task_id + 1
            # if router can be trained
            if router_enable:
                router_logits = router(hidden_states)[:, :end]
                if task_id is not None:
                    self.router_loss = F.cross_entropy(router_logits, torch.tensor([task_id] * B, device=x.device), reduction="mean")
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

            elif task_id is not None:
                routing_weights = F.one_hot(
                    torch.full((B,), task_id, device=x.device),
                    num_classes=num_tasks
                ).float()
                top_k = 1
            else:
                raise NotImplementedError

            # just use previous task id
            # print(end)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            routing_weights = routing_weights.to(dtype) #batch_size x top_k

            final_hidden_states = torch.zeros_like(x)
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_tasks).permute(2, 1, 0)
            # expert_mask: num_tasks x top_k x batch_size
            for i in range(num_tasks):
                idx, top_x = torch.where(expert_mask[i]) # top_x: mini_batch
                if top_x.numel() == 0:
                    continue 

                current_state = x.index_select(0, top_x) # mini_batch x sequence_length x hidden_dim
                if not self.use_dora[active_adapter]:
                    def lora_fn(state):
                        return lora_B[i](lora_A[i](state))

                    out = lora_fn(current_state) * (routing_weights.index_select(0, top_x).gather(1, idx[:, None]))[:, None, :]
                    final_hidden_states.index_add_(0, top_x, out)
                else:
                    base_result = result if isinstance(dropout, nn.Identity) or not self.training else None

                    def dora_fn(state):
                        return self.lora_magnitude_vector[active_adapter][i](
                            state,
                            lora_A=lora_A[i],
                            lora_B=lora_B[i],
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                            base_result=base_result,
                        )

                    out = dora_fn(current_state) * routing_weights.index_select(0, top_x).gather(1, idx[:, None])
                    final_hidden_states.index_add_(0, top_x, out)
            result = result + final_hidden_states.view(B, S, H) * scaling

        return result.to(torch_result_dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loramoe." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraMoEConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    else:
        raise ValueError(
            f"Unsupported target module type: {type(target_base_layer)}. "
            "Only torch.nn.Embedding and torch.nn.Linear are supported."
        )

    return new_module
