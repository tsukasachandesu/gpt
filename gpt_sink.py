import math
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import contextlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# Use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model
def forward_fill_from_updates(update_values: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
    """
    直近の更新値を forward-fill（LOCF: last observation carried forward）する。

    - update_values: (B,T)  update_mask==True の位置に新しい値（False位置は不定でもOK）
    - update_mask:   (B,T)  bool
    - return:        (B,T)  forward-fill 後の値（最初に更新が無い区間は 0）
    """
    if update_values.ndim != 2 or update_mask.ndim != 2:
        raise ValueError("update_values and update_mask must be (B,T)")
    if update_values.shape != update_mask.shape:
        raise ValueError("shapes must match")

    update_mask = update_mask.to(dtype=torch.bool)

    B, T = update_values.shape
    device = update_values.device

    # (B,T) の時刻インデックス（expand は view なので実メモリは増えない）
    t = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, T)

    # 更新があった位置だけ時刻を残し、累積最大で「直近の更新位置」を得る
    set_idx = t.masked_fill(~update_mask, -1)
    last_idx = torch.cummax(set_idx, dim=1).values.clamp_min(0)

    # 更新値以外は 0 に落として、直近更新位置から gather
    updates = update_values.masked_fill(~update_mask, 0)
    return updates.gather(1, last_idx)


class SinusoidalEncoding1D(nn.Module):
    """
    1D sin/cos 位置エンコーディング。

    - idx: (...,) の整数テンソル -> (..., d_model)

    torch.compile / 長系列向けに、max_len を指定した場合は
    (max_len, d_model) のテーブルを初期化時に作り、forward は embedding lookup のみにする。
    """
    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        *,
        max_len: Optional[int] = None,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even (sin/cos pairs).")

        self.d_model = int(d_model)
        self.base = float(base)
        self.max_len = None if max_len is None else int(max_len)

        # (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=dtype) * (-math.log(self.base) / self.d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

        if self.max_len is None:
            # TorchDynamo が attribute の有無で分岐しないようにダミーを置く
            self.register_buffer("table", torch.empty(0, dtype=dtype), persistent=False)
        else:
            # (max_len, d_model) を事前計算
            pos = torch.arange(self.max_len, dtype=dtype).unsqueeze(1)      # (L,1)
            angle = pos * div_term.unsqueeze(0)                             # (L, d/2)
            table = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(self.max_len, self.d_model)
            self.register_buffer("table", table, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.max_len is not None:
            # idx: (...,) -> (..., d)
            return F.embedding(idx, self.table)

        # angle: (..., d/2)
        angle = idx.to(self.div_term.dtype).unsqueeze(-1) * self.div_term
        # (..., d)
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(angle.shape[:-1] + (self.d_model,))


class REMIPosPitchSinusoidalPE(nn.Module):
    """
    2軸（pos / pitch）を 2D とみなした sin/cos PE（連結 / concatenate）。

    - Pos軸: posトークン または pitchトークンにのみ付与（要求仕様）
    - Pitch軸: 音符系トークンにのみ付与（note_mask でゲート）

    NOTE:
      代表的な 2D sin/cos 実装（例: ViT/MAE 系の get_2d_sincos_pos_embed）は、
      各軸に embed_dim/2 を割り当てて 1D sin/cos を作り、最後に concat する。
      そのため d_model は通常 4 の倍数（= 各軸が偶数次元）を前提にする。
    """
    def __init__(
        self,
        d_model: int,
        *,
        pos_start: int,
        pos_size: int = 128,
        pitch_start: int,
        pitch_size: int = 32,
        bar_id: Optional[int] = None,      # Barトークンがあるなら pos を 0 にリセット（PEにbar軸は入れない）
        doc_id: Optional[int] = None,      # Docトークンがあるなら pos/pitch を 0 にリセット（注意マスクと整合）
        pos_idx_offset: int = 0,           # 0: 従来互換。1: 0を「無効」に予約（pos=0 と reset=0 の衝突回避）
        pitch_idx_offset: int = 0,         # 0: 従来互換。1: 0を「無効」に予約（pitch=0 と no-pitch の衝突回避）
        base: float = 10000.0,
        dropout: float = 0.0,
        normalize: bool = True,            # True: 有効次元数に応じて sqrt(d_model / active_dims) で正規化
        carry_pitch: bool = False,         # True: Pitchを forward-fill（音符属性トークンにも付与したい場合）
        reset_pitch_on_pos: bool = True,   # carry_pitch時、Pos/Bar/Docで pitch を 0 にリセット
    ):
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sin/cos concat (pos/pitch axes).")

        self.pos_start = int(pos_start)
        self.pos_size = int(pos_size)
        self.pitch_start = int(pitch_start)
        self.pitch_size = int(pitch_size)
        self.bar_id = None if bar_id is None else int(bar_id)
        self.doc_id = None if doc_id is None else int(doc_id)

        self.pos_idx_offset = int(pos_idx_offset)
        self.pitch_idx_offset = int(pitch_idx_offset)
        if self.pos_idx_offset < 0 or self.pitch_idx_offset < 0:
            raise ValueError("pos_idx_offset and pitch_idx_offset must be >= 0")

        self.normalize = bool(normalize)
        self.carry_pitch = bool(carry_pitch)
        self.reset_pitch_on_pos = bool(reset_pitch_on_pos)

        # 2D sin/cos: 各軸に d_model/2 を割り当てて concat
        self.d_pos = self.d_model // 2
        self.d_pitch = self.d_model - self.d_pos

        # インデックスが小さい（pos_size/pitch_size）前提なのでテーブル化して lookup にする
        # NOTE: pos/pitch の「0番」を無効（reset）に予約したい場合は offset を 1 にする。
        pos_table_len = self.pos_size + self.pos_idx_offset
        pitch_table_len = self.pitch_size + self.pitch_idx_offset
        self.enc_pos = SinusoidalEncoding1D(self.d_pos, base=base, max_len=pos_table_len)
        self.enc_pitch = SinusoidalEncoding1D(self.d_pitch, base=base, max_len=pitch_table_len)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        token_ids: torch.Tensor,                  # (B,T) long
        *,
        x: Optional[torch.Tensor] = None,         # (B,T,d_model) を渡すと加算して返す
        note_mask: Optional[torch.Tensor] = None  # (B,T) bool。省略時は pitchトークンのみ音符系とみなす
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be (B,T)")

        if token_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise ValueError("token_ids must be an integer tensor")

        B, T = token_ids.shape

        # --- トークン種別マスク ---
        pos_tok = (token_ids >= self.pos_start) & (token_ids < self.pos_start + self.pos_size)
        pitch_tok = (token_ids >= self.pitch_start) & (token_ids < self.pitch_start + self.pitch_size)

        if self.bar_id is None:
            bar_tok = token_ids.new_zeros((B, T), dtype=torch.bool)
        else:
            bar_tok = token_ids == self.bar_id

        if self.doc_id is None:
            doc_tok = token_ids.new_zeros((B, T), dtype=torch.bool)
        else:
            doc_tok = token_ids == self.doc_id

        # 音符系ゲート：未指定なら簡略化REMIとして pitchトークンのみを音符系扱い
        if note_mask is None:
            note_mask = pitch_tok
        else:
            if note_mask.shape != (B, T):
                raise ValueError("note_mask must be (B,T)")
            note_mask = note_mask.to(dtype=torch.bool)

        # --- Pos: Positionトークン（およびBar/Docによる0リセット）を forward-fill してインデックス化 ---
        # NOTE: token_ids が uint8 等の場合、(token_ids - pos_start) が underflow しても
        # pos_tok==False 位置は最終的に 0 に落ちるため実害は無いが、意図を明確にするため where を使用。
        pos_update_mask = pos_tok | bar_tok | doc_tok
        pos_update_val = torch.where(
            pos_tok,
            (token_ids - self.pos_start + self.pos_idx_offset),
            torch.zeros_like(token_ids),
        )
        pos_idx = forward_fill_from_updates(pos_update_val, pos_update_mask)    # (B,T)

        # --- Pitch: note_mask 付与の元になるピッチインデックス ---
        if self.carry_pitch:
            pitch_update_mask = pitch_tok | doc_tok
            if self.reset_pitch_on_pos:
                pitch_update_mask = pitch_update_mask | pos_tok | bar_tok

            pitch_update_val = torch.where(
                pitch_tok,
                (token_ids - self.pitch_start + self.pitch_idx_offset),
                torch.zeros_like(token_ids),
            )
            pitch_idx = forward_fill_from_updates(pitch_update_val, pitch_update_mask)
        else:
            pitch_idx = torch.where(
                pitch_tok,
                (token_ids - self.pitch_start + self.pitch_idx_offset),
                torch.zeros_like(token_ids),
            )

        # --- PE計算（軸ごとに独立に sin/cos を作り、concat） ---
        pe_pos = self.enc_pos(pos_idx)         # (B,T,d_pos)
        pe_pitch = self.enc_pitch(pitch_idx)   # (B,T,d_pitch)

        # Pos は pos_tok または pitch_tok にのみ付与（要求仕様）
        pos_gate = (pos_tok | pitch_tok).to(dtype=pe_pos.dtype).unsqueeze(-1)   # (B,T,1)
        # Pitch は音符系（note_mask）にのみ付与
        pitch_gate = note_mask.to(dtype=pe_pos.dtype).unsqueeze(-1)             # (B,T,1)

        pe_pos = pe_pos * pos_gate
        pe_pitch = pe_pitch * pitch_gate
        pe = torch.cat((pe_pos, pe_pitch), dim=-1)                              # (B,T,d_model)

        # トークンごとに「有効次元数」で正規化（片側だけ有効なときに振幅が落ちるのを補正）
        if self.normalize:
            active_dims = pos_gate.squeeze(-1) * float(self.d_pos) + pitch_gate.squeeze(-1) * float(self.d_pitch)
            # active_dims==0 は pe==0 なので 1 扱いでOK（0除算回避）
            pe = pe * torch.rsqrt((active_dims.clamp_min(1.0) / float(self.d_model))).unsqueeze(-1)

        pe = self.drop(pe)

        if x is None:
            return pe

        if x.shape != pe.shape:
            raise ValueError(f"x must be (B,T,d_model)=={tuple(pe.shape)}, got {tuple(x.shape)}")

        if pe.dtype != x.dtype:
            pe = pe.to(dtype=x.dtype)

        return x + pe

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(torch.nn.Module):

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary dim must be even, got {dim}")
        # inv_freq: (dim/2,)
        inv_freq = (1.0 / float(base)) ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Lazy cache (rebuilt on seq_len / device / dtype change)
        self.seq_len_cached: int = -1
        self._device_cached: Optional[torch.device] = None
        self._dtype_cached: Optional[torch.dtype] = None
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _build_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        # Keep trig in fp32, then cast to `dtype` for the actual RoPE application.
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        self.cos_cached = freqs.cos().to(dtype=dtype)
        self.sin_cached = freqs.sin().to(dtype=dtype)
        self.seq_len_cached = int(seq_len)
        self._device_cached = device
        self._dtype_cached = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = int(x.shape[1])
        dev = x.device
        dt = x.dtype

        if (
            self.cos_cached is None
            or self.sin_cached is None
            or self.seq_len_cached != seq_len
            or self._device_cached != dev
            or self._dtype_cached != dt
        ):
            self._build_cache(seq_len, device=dev, dtype=dt)

        cos = self.cos_cached[None, :, None, :]
        sin = self.sin_cached[None, :, None, :]
        return apply_rotary_emb(x, cos, sin)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply (chunked) RoPE to a tensor.

    This codebase represents RoPE pairs as the first/second halves of the head dimension,
    i.e. pairs are (x[..., i], x[..., i + head_dim/2]). We keep that layout to avoid
    unnecessarily breaking existing checkpoints.

    Args:
        x: (B, T, n_head, head_dim)
        cos/sin: (B, T, 1, head_dim/2) or broadcastable to it

    Returns:
        (B, T, n_head, head_dim)
    """
    x1, x2 = x.chunk(2, dim=3)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=3).type_as(x)


class AxialRotaryEmbedding4D(nn.Module):
    """4軸（index / bar / pos / pitch）の軸別（axial）RoPE を生成する.

    - `index`: 従来のトークン index (0..T-1)
    - `bar`  : 小節番号（bar token の累積）
    - `pos`  : 小節内位置（pos token を forward-fill、bar/doc でリセット）
    - `pitch`: ピッチ値（pitch token を forward-fill 可能）

    実装方針は、Vision 2D RoPE と同様に「各軸に head_dim の一部を割り当て、
    軸ごとに周波数を計算して連結する」(axial frequency) 方式。

    追加機能（互換性を壊さない範囲で拡張）:
      - pos/pitch の index に offset を入れて 0 を「無効（回転しない）」に予約可能
      - RoPE の線形スケーリング（position interpolation 相当）を任意で適用可能
      - 4軸で割り当てたペアの並べ方を block / interleave で切替可能
      - pitch 回転のゲート（pitch_tok のみ等）を切替可能
    """

    def __init__(
        self,
        head_dim: int,
        *,
        base: int = 10_000,
        axial_fractions: tuple[int, int, int, int] = (1, 1, 1, 1),
        pitch_start: int = 0,
        pitch_size: int = 128,
        pos_start: int = 129,
        pos_size: int = 32,
        bar_id: Optional[int] = 128,
        doc_id: Optional[int] = 163,
        carry_pitch: bool = True,
        reset_pitch_on_pos: bool = True,
        reset_pitch_on_bar: bool = True,
        # 0 を「軸無効」に予約するためのオフセット（0: 従来互換、1: 衝突回避）
        pos_idx_offset: int = 0,
        pitch_idx_offset: int = 0,
        # RoPE scaling（長文外挿用）。現状は "linear" のみサポート（inv_freq /= factor）。
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        rope_scaling_apply_axes: tuple[bool, bool, bool, bool] = (True, True, False, False),  # (index, bar, pos, pitch)
        # ペア（head_dim/2）の並べ方: "block"（従来） or "interleave"
        pair_layout: str = "block",
        # pitch 回転のゲート: "none" or "pitch_tok_only"
        pitch_gate: str = "none",
        # If >0, RoPE caches are preallocated up to this sequence length. If 0, caches grow on demand.
        max_seq_len: int = 0,
        # If not None, caches are stored in this dtype (e.g. torch.bfloat16) to avoid
        # float32->bf16 conversion overhead at runtime. Trig is still computed in fp32.
        cache_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if any(int(r) <= 0 for r in axial_fractions):
            raise ValueError(f"axial_fractions must be positive, got {axial_fractions}")

        self.head_dim = int(head_dim)
        self.base = int(base)
        self.pitch_start = int(pitch_start)
        self.pitch_size = int(pitch_size)
        self.pos_start = int(pos_start)
        self.pos_size = int(pos_size)
        self.bar_id = int(bar_id) if bar_id is not None else None
        self.doc_id = int(doc_id) if doc_id is not None else None
        self.carry_pitch = bool(carry_pitch)
        self.reset_pitch_on_pos = bool(reset_pitch_on_pos)
        self.reset_pitch_on_bar = bool(reset_pitch_on_bar)
        self.max_seq_len = int(max_seq_len)

        # Optional cache dtype (store cached cos/sin in bf16/f16 to avoid per-step casts).
        if cache_dtype is not None and cache_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "cache_dtype must be one of {float32, float16, bfloat16} or None, "
                f"got {cache_dtype!r}"
            )
        self.cache_dtype = cache_dtype

        # pos/pitch index offsets (reserve 0 for "axis disabled")
        self.pos_idx_offset = int(pos_idx_offset)
        self.pitch_idx_offset = int(pitch_idx_offset)
        if self.pos_idx_offset < 0 or self.pitch_idx_offset < 0:
            raise ValueError("pos_idx_offset and pitch_idx_offset must be >= 0")
        self.pos_cache_len = int(self.pos_size + self.pos_idx_offset)
        self.pitch_cache_len = int(self.pitch_size + self.pitch_idx_offset)

        # RoPE scaling config
        self.rope_scaling_type = None if rope_scaling_type is None else str(rope_scaling_type).lower()
        if self.rope_scaling_type in ("none", ""):
            self.rope_scaling_type = None
        self.rope_scaling_factor = float(rope_scaling_factor)
        if self.rope_scaling_type is not None:
            if self.rope_scaling_factor <= 0:
                raise ValueError(f"rope_scaling_factor must be > 0, got {rope_scaling_factor}")
            if self.rope_scaling_type not in ("linear",):
                raise ValueError(f"Unsupported rope_scaling_type={rope_scaling_type!r}. Supported: None, 'linear'.")

        if len(rope_scaling_apply_axes) != 4:
            raise ValueError("rope_scaling_apply_axes must be a 4-tuple: (index, bar, pos, pitch)")
        self.rope_scaling_apply_axes = tuple(bool(x) for x in rope_scaling_apply_axes)

        # pair layout + pitch gate
        self.pair_layout = str(pair_layout).lower()
        if self.pair_layout not in ("block", "interleave"):
            raise ValueError(f"pair_layout must be 'block' or 'interleave', got {pair_layout!r}")

        self.pitch_gate = str(pitch_gate).lower()
        if self.pitch_gate not in ("none", "pitch_tok_only"):
            raise ValueError(f"pitch_gate must be 'none' or 'pitch_tok_only', got {pitch_gate!r}")

        # Split head_dim/2 "pairs" across the 4 axes.
        total_pairs = head_dim // 2
        pairs_index, pairs_bar, pairs_pos, pairs_pitch = self._allocate_pairs(total_pairs, axial_fractions)
        if min(pairs_index, pairs_bar, pairs_pos, pairs_pitch) <= 0:
            raise ValueError(
                "All axes must receive at least 1 pair. "
                f"head_dim={head_dim} -> total_pairs={total_pairs}, "
                f"axial_fractions={axial_fractions} -> pairs="
                f"(index={pairs_index}, bar={pairs_bar}, pos={pairs_pos}, pitch={pairs_pitch})"
            )
        self.pairs_index = int(pairs_index)
        self.pairs_bar = int(pairs_bar)
        self.pairs_pos = int(pairs_pos)
        self.pairs_pitch = int(pairs_pitch)

        # Precompute inverse frequencies (inv_freq) per axis.
        theta_index = self._build_theta(self.pairs_index * 2, base=self.base)
        theta_bar = self._build_theta(self.pairs_bar * 2, base=self.base)
        theta_pos = self._build_theta(self.pairs_pos * 2, base=self.base)
        theta_pitch = self._build_theta(self.pairs_pitch * 2, base=self.base)

        theta_index = self._apply_rope_scaling(theta_index, apply=self.rope_scaling_apply_axes[0])
        theta_bar = self._apply_rope_scaling(theta_bar, apply=self.rope_scaling_apply_axes[1])
        theta_pos = self._apply_rope_scaling(theta_pos, apply=self.rope_scaling_apply_axes[2])
        theta_pitch = self._apply_rope_scaling(theta_pitch, apply=self.rope_scaling_apply_axes[3])

        self.register_buffer("theta_index", theta_index, persistent=False)
        self.register_buffer("theta_bar", theta_bar, persistent=False)
        self.register_buffer("theta_pos", theta_pos, persistent=False)
        self.register_buffer("theta_pitch", theta_pitch, persistent=False)

        # Permutation for pair layout (applied after concatenation).
        pair_perm = self._build_pair_perm(
            total_pairs=total_pairs,
            pairs=(self.pairs_index, self.pairs_bar, self.pairs_pos, self.pairs_pitch),
            layout=self.pair_layout,
        )
        self.register_buffer("pair_perm", pair_perm, persistent=False)
        # Small runtime optimization: skip index_select when identity.
        self._pair_perm_is_identity = bool(torch.equal(pair_perm.cpu(), torch.arange(total_pairs, dtype=torch.long)))

        # Caches (computed lazily on the right device).
        self._cos_index = None
        self._sin_index = None
        self._cos_bar = None
        self._sin_bar = None
        self._cos_pos = None
        self._sin_pos = None
        self._cos_pitch = None
        self._sin_pitch = None

    def _apply_rope_scaling(self, inv_freq: torch.Tensor, *, apply: bool) -> torch.Tensor:
        """Apply RoPE scaling to inverse frequencies.

        - linear scaling: inv_freq /= factor
          (HF Transformers の position interpolation と同等の変形)。
        """
        if (not apply) or self.rope_scaling_type is None or self.rope_scaling_factor == 1.0:
            return inv_freq
        if self.rope_scaling_type == "linear":
            return inv_freq / float(self.rope_scaling_factor)
        # __init__ で弾くので通常到達しない
        raise RuntimeError(f"Unsupported rope_scaling_type={self.rope_scaling_type!r}")

    @staticmethod
    def _allocate_pairs(total_pairs: int, ratios: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Deterministically allocate `total_pairs` across 4 axes according to integer ratios."""
        r0, r1, r2, r3 = (int(r) for r in ratios)
        denom = r0 + r1 + r2 + r3
        scaled = [total_pairs * r0, total_pairs * r1, total_pairs * r2, total_pairs * r3]
        base = [s // denom for s in scaled]
        frac = [s % denom for s in scaled]
        rem = total_pairs - sum(base)
        # Distribute the remainder to the largest fractional parts (stable tie-break by axis order).
        order = sorted(range(4), key=lambda i: (-frac[i], i))
        for i in order[:rem]:
            base[i] += 1
        return base[0], base[1], base[2], base[3]

    @staticmethod
    def _build_theta(dim: int, *, base: int) -> torch.Tensor:
        # Standard RoPE geometric progression.
        # inv_freq shape: [dim/2]
        ar = torch.arange(0, dim, 2)[: (dim // 2)].float()
        return 1.0 / (float(base) ** (ar / float(dim)))

    @staticmethod
    def _build_pair_perm(*, total_pairs: int, pairs: tuple[int, int, int, int], layout: str) -> torch.Tensor:
        """Build permutation indices to map block-concat -> requested layout."""
        if layout == "block":
            return torch.arange(total_pairs, dtype=torch.long)

        # layout == "interleave": round-robin across axes (index, bar, pos, pitch)
        p0, p1, p2, p3 = (int(x) for x in pairs)
        offsets = (0, p0, p0 + p1, p0 + p1 + p2)
        remaining = [p0, p1, p2, p3]
        cursors = [0, 0, 0, 0]
        out: list[int] = []
        axis_order = (0, 1, 2, 3)
        while len(out) < total_pairs:
            for ax in axis_order:
                if cursors[ax] < remaining[ax]:
                    out.append(offsets[ax] + cursors[ax])
                    cursors[ax] += 1
                    if len(out) >= total_pairs:
                        break
        return torch.tensor(out, dtype=torch.long)

    @staticmethod
    def _build_cache(
        theta: torch.Tensor,
        max_len: int,
        *,
        device: torch.device,
        cache_dtype: Optional[torch.dtype],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (cos, sin) lookup tables.

        - Trig is computed in fp32 for numerical stability.
        - If `cache_dtype` is not None, the final tables are cast to that dtype to
          avoid per-step float32->(bf16/f16) conversion overhead.
        """
        seq = torch.arange(max_len, device=device, dtype=torch.float32)
        freqs = torch.outer(seq, theta.to(device=device, dtype=torch.float32))
        cos = freqs.cos()
        sin = freqs.sin()
        if cache_dtype is not None and cache_dtype != cos.dtype:
            cos = cos.to(dtype=cache_dtype)
            sin = sin.to(dtype=cache_dtype)
        return cos, sin

    @torch.no_grad()
    def _maybe_init_caches(self, *, device: torch.device, seq_len: int) -> None:
        # index cache (0..seq_len-1)
        need_index_len = max(seq_len, self.max_seq_len)
        if (
            self._cos_index is None
            or self._cos_index.device != device
            or self._cos_index.size(0) < need_index_len
        ):
            self._cos_index, self._sin_index = self._build_cache(
                self.theta_index,
                need_index_len,
                device=device,
                cache_dtype=self.cache_dtype,
            )

        # bar cache: bar_idx is a cumulative count, safe upper bound is seq_len (one bar token per step).
        need_bar_len = max(seq_len + 1, (self.max_seq_len + 1) if self.max_seq_len > 0 else 0)
        if (
            self._cos_bar is None
            or self._cos_bar.device != device
            or self._cos_bar.size(0) < need_bar_len
        ):
            self._cos_bar, self._sin_bar = self._build_cache(
                self.theta_bar,
                need_bar_len,
                device=device,
                cache_dtype=self.cache_dtype,
            )

        # pos cache: fixed (0..pos_size-1) or (0..pos_size) when offset==1
        if (
            self._cos_pos is None
            or self._cos_pos.device != device
            or self._cos_pos.size(0) != self.pos_cache_len
        ):
            self._cos_pos, self._sin_pos = self._build_cache(
                self.theta_pos,
                self.pos_cache_len,
                device=device,
                cache_dtype=self.cache_dtype,
            )

        # pitch cache: fixed (0..pitch_size-1) or (0..pitch_size) when offset==1
        if (
            self._cos_pitch is None
            or self._cos_pitch.device != device
            or self._cos_pitch.size(0) != self.pitch_cache_len
        ):
            self._cos_pitch, self._sin_pitch = self._build_cache(
                self.theta_pitch,
                self.pitch_cache_len,
                device=device,
                cache_dtype=self.cache_dtype,
            )

    @torch.no_grad()
    def precompute_caches(self, seq_len: int, *, device: Optional[torch.device] = None) -> None:
        """Optional warm-up to avoid lazy cache construction inside torch.compile graphs."""
        if device is None:
            device = self.theta_index.device
        self._maybe_init_caches(device=device, seq_len=int(seq_len))

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        note_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: (B,T) or (T,)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be 1D or 2D, got shape={tuple(token_ids.shape)}")
        B, T = token_ids.shape
        device = token_ids.device
        self._maybe_init_caches(device=device, seq_len=T)

        # --- axis indices ---
        # masks
        pitch_tok = (token_ids >= self.pitch_start) & (token_ids < (self.pitch_start + self.pitch_size))
        pos_tok = (token_ids >= self.pos_start) & (token_ids < (self.pos_start + self.pos_size))
        bar_tok = token_ids.eq(self.bar_id) if self.bar_id is not None else torch.zeros_like(token_ids, dtype=torch.bool)
        doc_tok = token_ids.eq(self.doc_id) if self.doc_id is not None else torch.zeros_like(token_ids, dtype=torch.bool)

        # bar index (cumulative count of bar tokens)
        bar_idx = torch.cumsum(bar_tok.to(dtype=torch.long), dim=1)
        # NOTE: document boundaries are already masked in attention, so a doc-wise reset is optional.
        # We keep it to make the semantics ("bar number") cleaner.
        if self.doc_id is not None:
            bar_offset = forward_fill_from_updates(bar_idx, doc_tok)
            bar_idx = bar_idx - bar_offset

        # pos index (forward-fill from pos tokens; reset to 0 on bar/doc)
        pos_updates = torch.where(
            pos_tok,
            (token_ids - self.pos_start + self.pos_idx_offset),
            torch.zeros_like(token_ids),
        )
        pos_update_mask = pos_tok | bar_tok | doc_tok
        pos_idx = forward_fill_from_updates(pos_updates, pos_update_mask)

        # pitch index (optional forward-fill; reset on pos/bar/doc)
        pitch_updates = torch.where(
            pitch_tok,
            (token_ids - self.pitch_start + self.pitch_idx_offset),
            torch.zeros_like(token_ids),
        )
        if self.carry_pitch:
            pitch_update_mask = pitch_tok | doc_tok
            if self.reset_pitch_on_pos:
                pitch_update_mask = pitch_update_mask | pos_tok
            if self.reset_pitch_on_bar:
                pitch_update_mask = pitch_update_mask | bar_tok
            pitch_idx = forward_fill_from_updates(pitch_updates, pitch_update_mask)
        else:
            pitch_idx = pitch_updates

        # Optional gating for pitch rotations
        eff_note_mask = None
        if note_mask is not None:
            if note_mask.ndim == 1:
                note_mask = note_mask.unsqueeze(0)
            if note_mask.shape != (B, T):
                raise ValueError("note_mask must be (B,T) or (T,)")
            eff_note_mask = note_mask.to(dtype=torch.bool)
        else:
            if self.pitch_gate == "pitch_tok_only":
                eff_note_mask = pitch_tok

        if eff_note_mask is not None:
            pitch_idx = pitch_idx.masked_fill(~eff_note_mask, 0)

        # Clamp to cache size for safety (should be no-op if max_seq_len is configured sanely).
        bar_idx = bar_idx.clamp_max(self._cos_bar.size(0) - 1)
        pos_idx = pos_idx.clamp_max(self.pos_cache_len - 1)
        pitch_idx = pitch_idx.clamp_max(self.pitch_cache_len - 1)

        # --- gather cos/sin per axis ---
        cos_index = self._cos_index[:T].unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        sin_index = self._sin_index[:T].unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)

        cos_bar = F.embedding(bar_idx, self._cos_bar).unsqueeze(2)
        sin_bar = F.embedding(bar_idx, self._sin_bar).unsqueeze(2)
        cos_pos = F.embedding(pos_idx, self._cos_pos).unsqueeze(2)
        sin_pos = F.embedding(pos_idx, self._sin_pos).unsqueeze(2)
        cos_pitch = F.embedding(pitch_idx, self._cos_pitch).unsqueeze(2)
        sin_pitch = F.embedding(pitch_idx, self._sin_pitch).unsqueeze(2)

        # Concatenate along the "pairs" dimension to form full RoPE.
        cos = torch.cat((cos_index, cos_bar, cos_pos, cos_pitch), dim=-1)
        sin = torch.cat((sin_index, sin_bar, sin_pos, sin_pitch), dim=-1)

        # Optional permute pairs (layout change)
        if not self._pair_perm_is_identity:
            cos = cos.index_select(dim=-1, index=self.pair_perm)
            sin = sin.index_select(dim=-1, index=self.pair_perm)

        # Sanity check (runtime, but cheap) to catch misconfigurations early.
        if cos.size(-1) != (self.head_dim // 2):
            raise RuntimeError(
                f"AxialRoPE produced wrong last-dim: got {cos.size(-1)}, expected {self.head_dim // 2}. "
                f"pairs=(index={self.pairs_index},bar={self.pairs_bar},pos={self.pairs_pos},pitch={self.pairs_pitch})"
            )

        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, n_head, *, use_sink_bias: bool = False, sink_init: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.use_sink_bias = bool(use_sink_bias)
        if self.use_sink_bias:
            # GPT-OSS-style attention sink bias (a per-head 'virtual' sink logit).
            # Implemented via post-renormalization using FlexAttention's log-sum-exp (LSE).
            self.sink_weights = nn.Parameter(torch.full((n_head,), float(sink_init)))
        else:
            self.register_parameter("sink_weights", None)
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # value residual lambda
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977
        # rotary embeddings
        self.rotary = Rotary(dim // n_head) # dim // n_head = head_dim
        # output projection
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, vi, block_mask, rope_cos=None, rope_sin=None):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        v = (1 - self.lamb) * v + self.lamb * vi.view_as(v) # @Grad62304977
        q, k = norm(q), norm(k) # QK norm suggested by @Grad62304977
        if rope_cos is None or rope_sin is None:
            q, k = self.rotary(q), self.rotary(k)
        else:
            q, k = apply_rotary_emb(q, rope_cos, rope_sin), apply_rotary_emb(k, rope_cos, rope_sin)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if self.sink_weights is None:
            y = flex_attention(q_t, k_t, v_t, block_mask=block_mask)
        else:
            y, lse = flex_attention(q_t, k_t, v_t, block_mask=block_mask, return_lse=True)
            scale = torch.sigmoid(lse - self.sink_weights.view(1, -1, 1)).unsqueeze(-1)  # (B,H,T,1)
            y = (y * scale).to(q_t.dtype)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(
            config.n_embd,
            config.n_head,
            use_sink_bias=getattr(config, "use_attention_sink_bias", False),
            sink_init=getattr(config, "attention_sink_bias_init", 0.0),
        )
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask, rope_cos=None, rope_sin=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 164
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

    # Optional additive sinusoidal PE over (pos,pitch) for REMI-style tokens.
    use_remi_pe : bool = False

    pitch_start : int = 0
    pitch_size : int = 128
    pos_start : int = 129
    pos_size : int = 32
    bar_id : Optional[int] = 128

    # --- Axial RoPE (index / bar / pos / pitch) ---
    # Enable/disable axial RoPE. When disabled, attention falls back to legacy 1D RoPE over token index.
    use_axial_rope : bool = True

    # RoPE theta/base (many recent long-context models use much larger values; tune as needed).
    rope_base : int = 10_000

    # Allocate head_dim/2 "pairs" across the 4 axes proportionally.
    # Example: (1,1,1,1) splits equally; (2,1,1,0) is invalid (0 not allowed).
    rope_axial_fractions : tuple[int, int, int, int] = (1, 1, 1, 1)

    # Whether to forward-fill pitch across note attribute tokens (reset on pos/bar by default).
    rope_carry_pitch : bool = True
    rope_reset_pitch_on_pos : bool = True
    rope_reset_pitch_on_bar : bool = True

    # Document boundary token id (used both in attention masking and optional RoPE resets).
    rope_doc_id : Optional[int] = 163

    # If >0, RoPE caches are preallocated up to this sequence length.
    # If 0, caches grow on demand.
    rope_max_seq_len : int = 0

    # Reserve 0 index as "axis disabled" for pos/pitch (recommended: 1 to avoid collisions with valid 0 values).
    rope_pos_idx_offset : int = 1
    rope_pitch_idx_offset : int = 1

    # Optional RoPE scaling (long-context). Supported: None, "linear".
    # "linear" corresponds to position interpolation (inv_freq /= factor).
    rope_scaling_type : Optional[str] = None
    rope_scaling_factor : float = 1.0
    # Apply scaling to (index, bar, pos, pitch) axes.
    rope_scaling_apply_axes : tuple[bool, bool, bool, bool] = (True, True, False, False)

    # Pair layout within head_dim/2: "block" (default) or "interleave".
    rope_pair_layout : str = "block"

    # Gate pitch rotations: "none" or "pitch_tok_only".
    rope_pitch_gate : str = "none"

    # --- Attention Sink Bias (GPT-OSS-style: per-head 'virtual' sink logit) ---
    # If True, apply sink-bias renormalization using FlexAttention's returned LSE:
    #   out <- out * sigmoid(lse - sink_weight[h])
    # See e.g. Hugging Face's GPT-OSS notes and NeMo's reference implementation.
    use_attention_sink_bias : bool = False
    attention_sink_bias_init : float = 0.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
            vte = nn.Embedding(config.vocab_size, config.n_embd*12),
            # 4D axial RoPE (index / bar / pos / pitch). Shared across all attention layers.
            axial_rope = AxialRotaryEmbedding4D(
                head_dim=config.n_embd // config.n_head,
                base=config.rope_base,
                axial_fractions=config.rope_axial_fractions,
                pitch_start=config.pitch_start,
                pitch_size=config.pitch_size,
                pos_start=config.pos_start,
                pos_size=config.pos_size,
                bar_id=config.bar_id,
                doc_id=config.rope_doc_id,
                carry_pitch=config.rope_carry_pitch,
                reset_pitch_on_pos=config.rope_reset_pitch_on_pos,
                reset_pitch_on_bar=config.rope_reset_pitch_on_bar,
                pos_idx_offset=config.rope_pos_idx_offset,
                pitch_idx_offset=config.rope_pitch_idx_offset,
                rope_scaling_type=config.rope_scaling_type,
                rope_scaling_factor=config.rope_scaling_factor,
                rope_scaling_apply_axes=config.rope_scaling_apply_axes,
                pair_layout=config.rope_pair_layout,
                pitch_gate=config.rope_pitch_gate,
                max_seq_len=config.rope_max_seq_len,
            ),
            remi_pe = REMIPosPitchSinusoidalPE(
                config.n_embd,
                pos_start=config.pos_start,
                pos_size=config.pos_size,
                pitch_start=config.pitch_start,
                pitch_size=config.pitch_size,
                bar_id=config.bar_id,
                doc_id=config.rope_doc_id,
                pos_idx_offset=config.rope_pos_idx_offset,
                pitch_idx_offset=config.rope_pitch_idx_offset,
            ),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target, attn_blocksize):

        S = len(idx)

        # ---------------------------------------------------------------------
        # FlexAttention BlockMask: document + causal + sliding window
        #
        # - document masking (packed sequences): tokens can only attend within the same document
        # - sliding window causal: tokens can attend to at most `attn_blocksize` recent tokens
        # ---------------------------------------------------------------------
        doc_id = getattr(self.config, "rope_doc_id", None)
        if doc_id is None:
            doc_tok = idx.new_zeros((S,), dtype=torch.bool)
        else:
            doc_tok = idx.eq(int(doc_id))
        docs = doc_tok.cumsum(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask

        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device=idx.device, _compile=False)

        # forward the GPT model itself
        x = self.transformer.wte(idx[None]) # token embeddings of shape (b, t, n_embd)
        # Precompute (once per forward) RoPE cos/sin cache for all transformer blocks.
        rope_cos = rope_sin = None
        if getattr(self.config, "use_axial_rope", False):
            rope_cos, rope_sin = self.transformer.axial_rope(idx, dtype=x.dtype)
        if self.config.use_remi_pe:
            pe = self.transformer.remi_pe(idx[None]).to(dtype=x.dtype)
            x = x + pe
        x = norm(x) # @Grad62304977
        x0 = x
        vte = self.transformer.vte(idx[None])
        if self.config.use_remi_pe:
            vte = vte.view(1, S, 12, self.config.n_embd) + pe.unsqueeze(2)
            vi = vte.unbind(dim=2)
        else:
            vi = vte.chunk(12, dim=-1)

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, vi[i], x0, block_mask, rope_cos=rope_cos, rope_sin=rope_sin)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](
                x,
                vi[self.num_encoder_layers + i],
                x0,
                block_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'train.bin' # input .bin to train on
    input_val_bin : str = 'val.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8 # batch size, in sequences, across all devices
    sequence_length : int = 2048 * 32   # sequence length, in tokens
    num_iterations : int = 500 # number of iterations to run
    warmup_iters : int = 0
    cooldown_iters : int = 100 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 100 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 2048 * 32 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write(code)
        f.write('='*100 + '\n')
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# convenience variables
T = args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (T * ddp_world_size) == 0
val_steps = args.val_tokens // (T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size

# load tokens
train_loader = DistributedDataLoader(args.input_bin, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 164
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768, rope_max_seq_len=T))
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee

# Warm-up RoPE caches *before* torch.compile to avoid graph breaks from lazy cache init.
if hasattr(model, "transformer") and hasattr(model.transformer, "axial_rope"):
    try:
        # Store caches in bf16 to avoid per-step fp32->bf16 conversion.
        model.transformer.axial_rope.cache_dtype = torch.bfloat16
        model.transformer.axial_rope.precompute_caches(seq_len=T, device=torch.device(device))
    except Exception as e:
        # Fall back silently: this is an optimization path only.
        print0(f"[warn] axial_rope precompute skipped: {e}")
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight, raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
    attn_blocksize = torch.tensor(64*((step/args.num_iterations * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        ctx = model.no_sync() if i < train_accumulation_steps else contextlib.nullcontext()
        with ctx: # there's no need to sync gradients every accumulation step
            # forward pass
            loss = model(x, y, attn_blocksize=attn_blocksize)
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()
        train_loss = loss.detach()
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/300, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
