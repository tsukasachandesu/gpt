import torch
import torch.nn as nn
from typing import Optional

class FundamentalMusicEmbeddingSlots(nn.Module):
    """
    FME: sin/cos + bias + (Option) learnable_scale
    """
    def __init__(
        self,
        d_model: int,
        n_slots: int = 12,
        base: float = 10000.0,
        bias_init: str = "zeros",   # "zeros" or "normal"
        bias_std: float = 0.02,
        learnable_chunk_scale: bool = False, # <--- 追加
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairs (got {d_model}).")

        self.d_model = d_model
        self.n_slots = n_slots
        self.use_scale = learnable_chunk_scale

        i = torch.arange(d_model, dtype=torch.float32)
        exponent = (2.0 * torch.div(i, 2, rounding_mode="floor")) / float(d_model)
        angle_rates = torch.pow(torch.tensor(base, dtype=torch.float32), -exponent)
        self.register_buffer("angle_rates", angle_rates, persistent=True)

        # 1. Bias
        if bias_init == "zeros":
            bias = torch.zeros(n_slots, d_model, dtype=torch.float32)
        elif bias_init == "normal":
            bias = torch.randn(n_slots, d_model, dtype=torch.float32) * bias_std
        else:
            raise ValueError("bias_init must be 'zeros' or 'normal'")
        self.translation_bias = nn.Parameter(bias)

        # 2. Learnable Chunk Scale (追加部分)
        if self.use_scale:
            # 各スロットに対して1つのスカラー倍率を持つ (n_slots, 1)
            # 初期値は 1.0 にすることで学習初期の挙動を破壊しないようにする
            self.chunk_scale = nn.Parameter(torch.ones(n_slots, 1, dtype=torch.float32))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: (N,)
        x = inp.to(dtype=self.angle_rates.dtype)
        angle_rads = x.unsqueeze(-1) * self.angle_rates  # (N, d_model)

        base_enc = torch.empty_like(angle_rads)
        base_enc[..., 0::2] = torch.sin(angle_rads[..., 0::2])
        base_enc[..., 1::2] = torch.cos(angle_rads[..., 1::2])

        # (N, 1, d) -> expand -> (N, n_slots, d)
        # バイアス加算: (N, n_slots, d) + (1, n_slots, d)
        out = base_enc.unsqueeze(1).expand(-1, self.n_slots, -1) + self.translation_bias.unsqueeze(0)

        # Scale適用 (追加部分)
        if self.use_scale:
            # (N, n_slots, d) * (1, n_slots, 1)
            out = out * self.chunk_scale.unsqueeze(0)

        return out.reshape(x.shape[0], self.n_slots * self.d_model)


class MusicVTE_FMEFast(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_slots: int = 12,
        pitch_start: int = 0,
        pitch_size: int = 128,
        bar_start: int = 128,
        bar_size: int = 1,
        pos_start: int = 129,
        pos_size: int = 32,
        padding_idx: Optional[int] = None,
        pitch_base: float = 10000.0,
        bar_base: float = 10000.0,
        pos_base: float = 10000.0,
        fme_bias_init: str = "zeros",
        fme_bias_std: float = 0.02,
        learnable_chunk_scale: bool = False, # <--- 追加
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_slots = n_slots

        self.pitch_start, self.pitch_size = pitch_start, pitch_size
        self.bar_start, self.bar_size = bar_start, bar_size
        self.pos_start, self.pos_size = pos_start, pos_size

        self.token_embed = nn.Embedding(vocab_size, d_model * n_slots, padding_idx=padding_idx)

        # 各FMEモジュールへフラグを渡す
        self.pitch_fme = FundamentalMusicEmbeddingSlots(
            d_model=d_model, n_slots=n_slots, base=pitch_base,
            bias_init=fme_bias_init, bias_std=fme_bias_std,
            learnable_chunk_scale=learnable_chunk_scale
        )
        self.bar_fme = FundamentalMusicEmbeddingSlots(
            d_model=d_model, n_slots=n_slots, base=bar_base,
            bias_init=fme_bias_init, bias_std=fme_bias_std,
            learnable_chunk_scale=learnable_chunk_scale
        )
        self.pos_fme = FundamentalMusicEmbeddingSlots(
            d_model=d_model, n_slots=n_slots, base=pos_base,
            bias_init=fme_bias_init, bias_std=fme_bias_std,
            learnable_chunk_scale=learnable_chunk_scale
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # 既存のロジックと同じ
        if idx.dtype != torch.long:
            idx = idx.long()
        if idx.dim() < 1:
            raise ValueError(f"idx must have at least 1 dim, got shape={tuple(idx.shape)}")

        *prefix, T = idx.shape
        idx2 = idx.reshape(-1, T)

        emb = self.token_embed(idx2)
        out_dtype = emb.dtype

        pitch_mask = (idx2 >= self.pitch_start) & (idx2 < self.pitch_start + self.pitch_size)
        pos_mask   = (idx2 >= self.pos_start)   & (idx2 < self.pos_start + self.pos_size)
        bar_mask   = (idx2 >= self.bar_start)   & (idx2 < self.bar_start + self.bar_size)

        if pitch_mask.any().item():
            pitch_val = (idx2[pitch_mask] - self.pitch_start).to(torch.float32)
            pitch_emb = self.pitch_fme(pitch_val).to(dtype=out_dtype)
            emb[pitch_mask] = pitch_emb

        if pos_mask.any().item():
            pos_val = (idx2[pos_mask] - self.pos_start).to(torch.float32)
            pos_emb = self.pos_fme(pos_val).to(dtype=out_dtype)
            emb[pos_mask] = pos_emb

        if bar_mask.any().item():
            if self.bar_size == 1:
                bar_index = torch.cumsum(bar_mask.to(torch.long), dim=1) - 1
                bar_val = bar_index[bar_mask].to(torch.float32)
            else:
                bar_val = (idx2[bar_mask] - self.bar_start).to(torch.float32)

            bar_emb = self.bar_fme(bar_val).to(dtype=out_dtype)
            emb[bar_mask] = bar_emb

        return emb.reshape(*prefix, T, self.d_model * self.n_slots)

