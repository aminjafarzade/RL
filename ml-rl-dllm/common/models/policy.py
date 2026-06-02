#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from common.models.modeling_llada import LLaDABlock
from common.models.policy_layers import RoPEDiTBlock
from common.models.policy_layers import sinusoidal_time_embedding


class PolicyConfig(PretrainedConfig):
    model_type = "policy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PolicyHFWrapper(PreTrainedModel):
    config_class = PolicyConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_policy: nn.Module,
        policy_type: str,
        config: PolicyConfig | None = None,
    ):
        super().__init__(config or PolicyConfig())
        self.base_policy = base_policy
        self.policy_type = policy_type

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self._propagate_gc_flag(True)

        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        self._propagate_gc_flag(False)

    def _propagate_gc_flag(self, value: bool):
        if hasattr(self.base_policy, "set_gradient_checkpointing"):
            self.base_policy.set_gradient_checkpointing(value)
        else:
            setattr(self.base_policy, "gradient_checkpointing", bool(value))
        for m in self.base_policy.modules():
            if hasattr(m, "set_gradient_checkpointing"):
                m.set_gradient_checkpointing(value)
            elif hasattr(m, "gradient_checkpointing"):
                setattr(m, "gradient_checkpointing", bool(value))

    def forward(self, *args, **kwargs):
        # coerce dtypes if needed
        args = tuple(
            arg.to(self.dtype) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        kwargs = {
            k: v.to(self.dtype) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return self.base_policy(*args, **kwargs)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, *args, **kwargs):
        pass

    def tie_weights(self):
        pass


class DiTHiddenStatePolicy(nn.Module):
    def __init__(
        self,
        dllm,
        time_embed_dim: int = 128,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
    ):
        super().__init__()
        self.hidden_dim = dllm.config.hidden_size
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        # AdaLNs: since we attach an additional head (original llada block), we cannot
        # inject modulation WITHIN the block, as we do for DiTConfidencePolicy.
        # Instead, we inject scale+shift modulation (1) before each block, and (2) after the final block.
        # In the case of only having one block (as in the paper), this effectively only
        # moves one adaptation from before the FFN to after it, so it should offer similar performance.
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(num_blocks + 1)]
        )
        self.ada_linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
                for _ in range(num_blocks + 1)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LLaDABlock.build(i, dllm.model.config, dllm.model._LLaDAModel__cache)
                for i in range(num_blocks)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Initialize for controlled logit distribution.

        :param target_logit: Target value for initial logit mean
        """
        with torch.no_grad():
            for linear in self.ada_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            self.output_proj.bias.data.fill_(target_logit)

    def forward(
        self,
        m: torch.Tensor,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param hidden_states: (*B,L,hidden_dim) hidden states from DLLM
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :return: (*B,L) unmasking logits
        """
        *B, L, D = hidden_states.shape
        assert D == self.hidden_dim, f"Expected D={self.hidden_dim}, got {D=}"
        assert m.shape == (*B, L)
        assert isinstance(timestep, torch.Tensor)
        assert timestep.shape == (*B, 1), (
            f"Unexpected {timestep.shape=}; batch dim(s) {B=}"
        )

        x = hidden_states

        # Build conditioning: time + mask
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        )
        time_embed = time_embed.to(cond.dtype)
        time_embed = time_embed.expand((*([-1] * len(B)), L, -1))
        time_embed = self.time_mlp(time_embed)
        cond = cond + time_embed

        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)

        for i, block in enumerate(self.transformer_blocks):
            scale, bias = self.ada_linears[i](cond_flat).chunk(2, dim=-1)
            x_flat = self.norms[i](x_flat) * (1 + scale) + bias
            x_flat, _ = block(x_flat)

        scale, bias = self.ada_linears[-1](cond_flat).chunk(2, dim=-1)
        x_flat = self.norms[-1](x_flat) * (1 + scale) + bias

        x = x_flat.view(original_shape)
        raw_logits = self.output_proj(x).squeeze(-1)

        return raw_logits


class DiTConfidencePolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        feedforward_dim: int = 512,
        num_heads: int = 1,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        confidences_top_p: int = 1,
        extra_feature_dim: int = 0,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks
        self.confidences_top_p = confidences_top_p
        self.extra_feature_dim = extra_feature_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.confidence_proj = nn.Linear(
            confidences_top_p + extra_feature_dim, self.hidden_dim
        )
        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                RoPEDiTBlock(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Initialize for controlled logit distribution.

        Initialize mid-transformer AdaLNs to identities and sets output_proj bias to target_logit.
        This ensures initial logits are centered at target_logit, which is useful for
        DPLS sampling where you want to control the proportion of logits
        above/below the stop threshold.

        :param target_logit: Target value for initial logit mean (e.g., 0.0 to match stop_logit=0.0,
            or negative values to bias toward stopping earlier)
        """
        with torch.no_grad():
            # Initialize ada_lns to identity: x_norm * (1 + 0) + 0 = x_norm
            for block in self.transformer_blocks:
                block.ada_conditioning.weight.data.zero_()
                block.ada_conditioning.bias.data.zero_()

            # Initialize output_proj bias to target mean
            self.output_proj.bias.data.fill_(target_logit)

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,confidences_top_p + extra_feature_dim) confidence values and optional features
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :return: (*B,L) unmasking logits
        """
        *B, L, _ = c.shape
        assert m.shape == (*B, L)
        assert isinstance(timestep, torch.Tensor)
        assert timestep.shape == (*B, 1), (
            f"Unexpected {timestep.shape=}; batch dim(s) {B=}"
        )

        # Conditioning: time + mask
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        )
        time_embed = time_embed.to(cond.dtype)
        time_embed = time_embed.expand((*([-1] * len(B)), L, -1))
        time_embed = self.time_mlp(time_embed)
        cond = cond + time_embed

        # Embed confidences
        x = self.confidence_proj(c)

        # Transformer
        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)
        for block in self.transformer_blocks:
            x_flat = block(x_flat, cond_flat)
        x = x_flat.view(original_shape)

        # Predict logits
        x = self.final_norm(x)
        raw_logits = self.output_proj(x).squeeze(-1)

        return raw_logits
