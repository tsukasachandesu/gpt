from __future__ import annotations

import os
import sys

with open(sys.argv[0], encoding="utf-8", errors="replace") as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from itertools import cycle
from pathlib import Path
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

# Choose a CUDA device for the tiny backward() warmup (prevents a bug on some systems).
_bootstrap_dev = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
if "--infer" in sys.argv:
    try:
        if "--device" in sys.argv:
            _bootstrap_dev = sys.argv[sys.argv.index("--device") + 1]
    except Exception:
        pass
if torch.cuda.is_available() and str(_bootstrap_dev).startswith("cuda"):
    torch.empty(1, device=_bootstrap_dev, requires_grad=True).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# FlexAttention is intended to run under torch.compile; keep a compiled callable.
# Set DISABLE_FLEX_ATTN_COMPILE=1 to force eager (mainly for debugging).
if os.environ.get("DISABLE_FLEX_ATTN_COMPILE", "0") == "1":
    _flex_attention_impl = flex_attention
else:
    _flex_attention_impl = torch.compile(flex_attention, dynamic=False)


def _flex_attention_call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return _flex_attention_impl(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )

if hasattr(dynamo.config, "recompile_limit"):
    dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Aggregated hyperparameters (model + REMI96/tokenizer)

@dataclass(frozen=True)
class Remi96Spec:
    pitch_min: int = 0
    pitch_max: int = 95
    bar_id: int = 96
    pos_offset: int = 97
    bar_ticks: int = 32  # pos_size for 4/4, tpq=8
    tpq: int = 8
    ts_num: int = 4
    ts_denom: int = 4

    @property
    def pos_size(self) -> int:
        return int(self.bar_ticks)

    @property
    def pitch_size(self) -> int:
        return int(self.pitch_max - self.pitch_min + 1)

    @property
    def bos_id(self) -> int:
        return int(self.pos_offset + self.pos_size)

    @property
    def eof_id(self) -> int:
        return int(self.bos_id + 1)

    @property
    def pad_id(self) -> int:
        return int(self.bos_id + 2)

    @property
    def vocab_size(self) -> int:
        return int(self.pad_id + 1)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_layers: int = 16
    num_heads: int = 8
    head_dim: int = 128
    model_dim: int = 1024


REMI96 = Remi96Spec()
MODEL_CFG = ModelConfig(vocab_size=REMI96.vocab_size)


# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

# -----------------------------------------------------------------------------
# Muon optimizer (from nanochat/muon.py)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update.
    Uses polar_express_coeffs defined above.
    """
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    if g.size(-2) > g.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if g.size(-2) > g.size(-1):
        X = X.mT
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz (Polar Express variant).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        params = list(params)
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')

    def reset(self):
        for group in self.param_groups:
            params = group.get('params', [])
            if not params:
                continue
            state = self.state.get(params[0], None)
            if not state:
                continue
            for key in ("momentum_buffer", "second_momentum_buffer"):
                if key in state:
                    state[key].zero_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            state = self.state[params[0]]
            num_params = len(params)
            shape, device, dtype = params[0].shape, params[0].device, params[0].dtype

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
            momentum_buffer = state["momentum_buffer"]

            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    state["second_momentum_buffer"] = torch.zeros(num_params, shape[-2], 1, dtype=dtype, device=device)
                else:
                    state["second_momentum_buffer"] = torch.zeros(num_params, 1, shape[-1], dtype=dtype, device=device)
            second_momentum_buffer = state["second_momentum_buffer"]
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            stacked_grads = torch.stack([p.grad for p in params])
            stacked_params = torch.stack(params)

            self._momentum_t.fill_(group["momentum"])
            self._beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
            self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._wd_t.fill_(group["weight_decay"])

            muon_step_fused(
                stacked_grads,
                stacked_params,
                momentum_buffer,
                second_momentum_buffer,
                self._momentum_t,
                self._lr_t,
                self._wd_t,
                self._beta2_t,
                group["ns_steps"],
                red_dim,
            )

            torch._foreach_copy_(params, list(stacked_params.unbind(0)))


class DistMuon(torch.optim.Optimizer):
    """
    Distributed version of the Muon optimizer.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5, beta2: float = 0.95, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        params = list(params)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            chunk_size = (len(group_params) + world_size - 1) // world_size
            if rank == 0:
                print(f"Muon: {len(group_params)} params of shape {shape}, chunk_size={chunk_size}")
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
        super().__init__(param_groups, defaults)
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')

    def reset(self):
        for group in self.param_groups:
            params = group.get('params', [])
            if not params:
                continue
            state = self.state.get(params[0], None)
            if not state:
                continue
            for key in ("momentum_buffer", "second_momentum_buffer"):
                if key in state:
                    state[key].zero_()

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * world_size
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype

            grad_stack = torch.stack([p.grad for p in params])
            stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
            stacked_grads[:len(params)].copy_(grad_stack)
            if len(params) < padded_num_params:
                stacked_grads[len(params):].zero_()

            grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            group_infos.append(dict(
                grad_chunk=grad_chunk,
                reduce_future=reduce_future,
                stacked_grads=stacked_grads,
            ))

        all_gather_futures = []
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            chunk_size = group["chunk_size"]
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype
            grad_chunk = info["grad_chunk"]

            start_idx = rank * chunk_size
            num_owned = min(chunk_size, max(0, len(params) - start_idx))

            state = self.state[params[0]]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
            momentum_buffer = state["momentum_buffer"]

            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    state["second_momentum_buffer"] = torch.zeros(chunk_size, shape[-2], 1, dtype=dtype, device=device)
                else:
                    state["second_momentum_buffer"] = torch.zeros(chunk_size, 1, shape[-1], dtype=dtype, device=device)
            second_momentum_buffer = state["second_momentum_buffer"]
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

            if num_owned > 0:
                owned_params = [params[start_idx + i] for i in range(num_owned)]
                stacked_owned_params = torch.stack(owned_params)

                owned_grads = grad_chunk[:num_owned]
                owned_momentum = momentum_buffer[:num_owned]
                owned_second_momentum = second_momentum_buffer[:num_owned]

                self._momentum_t.fill_(group["momentum"])
                self._beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
                self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
                self._wd_t.fill_(group["weight_decay"])

                muon_step_fused(
                    owned_grads,
                    stacked_owned_params,
                    owned_momentum,
                    owned_second_momentum,
                    self._momentum_t,
                    self._lr_t,
                    self._wd_t,
                    self._beta2_t,
                    group["ns_steps"],
                    red_dim,
                )

                updated_params[:num_owned].copy_(stacked_owned_params)

            if num_owned < chunk_size:
                updated_params[num_owned:].zero_()

            stacked_params = info["stacked_grads"]
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            all_gather_futures.append(dict(
                gather_future=gather_future,
                stacked_params=stacked_params,
                params=params,
            ))

        for info in all_gather_futures:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            params = info["params"]
            torch._foreach_copy_(params, list(stacked_params[:len(params)].unbind(0)))


# -----------------------------------------------------------------------------
# Distributed AdamW optimizer (from nanochat/adamw.py)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update.
    Uses 0-D CPU tensors to avoid recompilation when hyperparameters change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer with sharded optimizer states (ZeRO-2 style).
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Validate
        if rank == 0:
            for group in param_groups:
                assert isinstance(group, dict), "expecting param_groups to be a list of dicts"
                assert isinstance(group["params"], list), "expecting group['params'] to be a list of tensors"
                for p in group["params"]:
                    sliced = p.numel() >= 1024
                    print(f"AdamW: 1 param of shape {p.shape}, sliced={sliced}")
                    if sliced:
                        assert p.shape[0] % world_size == 0, f"First dim of parameter shape {p.shape} must be divisible by world size {world_size}"
        super().__init__(param_groups, defaults)
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_futures: list[torch.Future] = []
        gather_futures: list[torch.Future] = []
        grad_slices = []
        is_small = []

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                grad = p.grad
                if p.numel() < 1024:
                    is_small.append(True)
                    reduce_futures.append(
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
                    grad_slices.append(grad)
                else:
                    is_small.append(False)
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    reduce_futures.append(
                        dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
                    grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for p in params:
                reduce_futures[idx].wait()
                g_slice = grad_slices[idx]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]

                if is_small[idx]:
                    p_slice = p
                else:
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                eff_wd = wd * getattr(p, "wd_mul", 1.0)
                self._step_t.fill_(state["step"])
                self._lr_t.fill_(lr)
                self._beta1_t.fill_(beta1)
                self._beta2_t.fill_(beta2)
                self._eps_t.fill_(eps)
                self._wd_t.fill_(eff_wd)

                adamw_step_fused(
                    p_slice, g_slice, exp_avg, exp_avg_sq,
                    self._step_t, self._lr_t, self._beta1_t, self._beta2_t, self._eps_t, self._wd_t,
                )

                if not is_small[idx]:
                    gather_futures.append(
                        dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                    )
                idx += 1

        if gather_futures:
            torch.futures.collect_all(gather_futures).wait()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

# YaRN + 多軸 MRoPE (index / pos / pitch / interval_same / interval_prev)
#
# * Interleaved-MRoPE compatible (fixed stride overwrite; stride = number of axes)
# * half-truncate RoPE is NOT used here (full rotary frequencies)

def forward_fill_from_updates(updates: Tensor, update_mask: Tensor, idx: Tensor) -> Tensor:
    """Forward-fill (last observation carried forward).

    本コードの用途（MRoPE の pos/pitch/interval インデックス生成）に特化して、
    **1次元 (T,)** を高速に処理する。

    Args:
        updates: (T,) int tensor
            update_mask=True の位置に「新しい値」が入っている。
        update_mask: (T,) bool
        idx: (T,) int tensor
            0..T-1 のインデックス（呼び出し側で一度だけ作って使い回す）

    Returns:
        (T,) で、各時刻が直近の update 値に置き換わる。
        直近の update がまだ存在しない区間は 0。

    Notes:
        * torch.compile 安定（Python ループ無し）
        * gather の index は int64 が必要なので最後に変換する
    """
    # updates/update_mask は (T,) 前提
    upd_idx = torch.where(update_mask, idx, idx.new_full((), -1).expand_as(idx))
    last_idx = torch.cummax(upd_idx, dim=0)[0]
    last_idx_clamped = last_idx.clamp_min(0).to(dtype=torch.int64)
    out = updates.gather(0, last_idx_clamped)
    out = torch.where(last_idx >= 0, out, torch.zeros_like(out))
    return out


class Yarn(nn.Module):
    """YaRN + 多軸 MRoPE（index / pos / pitch_oct / pitch_pc / interval_same / interval_prev）.

    互換性目標:
      * Interleaved-MRoPE（固定ストライド上書き）に対応（pair_layout='interleave'）
      * trig（cos/sin）は forward では計算しない（キャッシュ参照のみ）
      * torch.compile に乗る（Python ループなし）

    仕様（整理: 6 分割）:
      * 軸は常に 6 分割:
          0) index (base)
          1) pos
          2) pitch_oct  = pitch // 12（オクターブ）
          3) pitch_pc   = pitch % 12（半音クラス）
          4) interval_same
          5) interval_prev
      * bar 軸は削除（bar トークンは pos/pitch の reset と skyline 検出の補助にのみ使用）
      * pitch は 0 を "non pitch" として予約し、pitch_idx_offset を足してキャッシュ参照する。
        pitch の分割（//12, %12）は固定（pitch_decompose=False は不可、pitch_decompose_mod は 12 固定）。
      * interval の定義（pitch_abs は 0..pitch_size-1）:
          - interval_same: 同一 POS 群の skyline（最高音; 最初の pitch）との差分
              diff_same = pitch_abs - skyline_pitch_abs  ∈ [-(pitch_size-1), 0]
              skyline 自身は diff_same = 0 だが non pitch (idx=0) とは別扱い（ivl_idx_offset により分離）
          - interval_prev: 1 つ前の skyline との差分
              diff_prev = pitch_abs - prev_skyline_abs ∈ [-(pitch_size-1), +(pitch_size-1)]
              曲最初（前 skyline 不在）は non pitch と同じ扱い（idx=0）

    注意:
      * axis enable を OFF にした軸は「上書きしない」ため、その軸に割り当てられたペアは base（index または identity）で回転する。
      * forward の計算量は v2 と同等（pos / pitch / interval_same / interval_prev の embedding lookup 回数は増えない）。
        pitch_oct と pitch_pc は同一 pitch キャッシュの列を別セクションとして上書きするだけ。
    """

    AXIS_INDEX = 0
    AXIS_POS = 1
    AXIS_PITCH_OCT = 2
    AXIS_PITCH_PC = 3
    AXIS_IVL_SAME = 4
    AXIS_IVL_PREV = 5

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: int = 10000,
        *,
        # token ID 設定（本コードベースのデフォルト語彙に合わせる）
        pitch_start: int = REMI96.pitch_min,
        pitch_size: int = REMI96.pitch_size,
        bar_id: int | None = REMI96.bar_id,
        pos_start: int = REMI96.pos_offset,
        pos_size: int = REMI96.pos_size,
        # doc 境界（BOS=129 が既定。None で無効化）
        doc_id: int | None = REMI96.bos_id,
        # pitch/pos の forward-fill と reset 条件
        carry_pitch: bool = True,
        reset_pitch_on_pos: bool = True,
        reset_pitch_on_bar: bool = True,
        # 0 を「軸無効」に予約したい場合のオフセット
        pos_idx_offset: int = 1,
        pitch_idx_offset: int = 1,
        # pitch 回転のゲート
        pitch_gate: str = 'pitch_tok_only',
        # pitch 分解は固定（互換のため引数は残すが False は許可しない）
        pitch_decompose: bool = True,
        pitch_decompose_mod: int = 12,
        # 互換: axial_fractions が 4 or 5 要素のときだけ pitch_total を (oct, pc) に割る比率として使用
        pitch_decompose_fractions: tuple[int, int] = (1, 1),
        # rotary pairs の割当（head_dim/2 を 6軸に分配）
        #   * 互換:
        #       - 6 要素: (index, pos, pitch_oct, pitch_pc, interval_same, interval_prev)
        #       - 5 要素: (index, pos, pitch_total, interval_same, interval_prev)
        #       - 4 要素: (index, bar, pos, pitch_total)  ※bar 比率は (interval_same, interval_prev) 合計として再利用（等分）
        axial_fractions: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
        # ペア並び: 'interleave'（固定ストライド上書き） or 'block'
        pair_layout: str = 'interleave',
        # 軸ごとの ON/OFF
        enable_index: bool = True,
        enable_pos: bool = True,
        enable_pitch: bool = True,         # pitch_oct と pitch_pc をまとめて制御
        enable_ivl_same: bool = True,
        enable_ivl_prev: bool = True,
        # interval 側の 0 予約（non pitch）用オフセット（>=1）
        ivl_idx_offset: int = 1,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.base = int(base)

        self.pitch_start = int(pitch_start)
        self.pitch_size = int(pitch_size)
        self.bar_id = int(bar_id) if bar_id is not None else None
        self.pos_start = int(pos_start)
        self.pos_size = int(pos_size)
        self.doc_id = int(doc_id) if doc_id is not None else None

        self.carry_pitch = bool(carry_pitch)
        self.reset_pitch_on_pos = bool(reset_pitch_on_pos)
        self.reset_pitch_on_bar = bool(reset_pitch_on_bar)

        self.pos_idx_offset = int(pos_idx_offset)
        self.pitch_idx_offset = int(pitch_idx_offset)
        if self.pos_idx_offset < 0 or self.pitch_idx_offset < 0:
            raise ValueError('pos_idx_offset and pitch_idx_offset must be >= 0')
        self.pos_cache_len = int(self.pos_size + self.pos_idx_offset)
        self.pitch_cache_len = int(self.pitch_size + self.pitch_idx_offset)

        self.ivl_idx_offset = int(ivl_idx_offset)
        if self.ivl_idx_offset <= 0:
            raise ValueError(f"ivl_idx_offset must be >= 1 (reserve 0 for non pitch), got {ivl_idx_offset!r}")

        # interval ranges (in semitone difference, based on pitch_size)
        # same-time skyline diff is in [-(pitch_size-1), 0]
        # prev-skyline diff is in [-(pitch_size-1), +(pitch_size-1)]
        self.ivl_same_min = int(-(self.pitch_size - 1))
        self.ivl_same_max = int(0)
        self.ivl_prev_min = int(-(self.pitch_size - 1))
        self.ivl_prev_max = int(self.pitch_size - 1)

        self.ivl_same_size = int(self.ivl_same_max - self.ivl_same_min + 1)
        self.ivl_prev_size = int(self.ivl_prev_max - self.ivl_prev_min + 1)

        self.ivl_same_cache_len = int(self.ivl_same_size + self.ivl_idx_offset)
        self.ivl_prev_cache_len = int(self.ivl_prev_size + self.ivl_idx_offset)

        self.pitch_gate = str(pitch_gate).lower()
        if self.pitch_gate not in ('none', 'pitch_tok_only'):
            raise ValueError(f"pitch_gate must be 'none' or 'pitch_tok_only', got {pitch_gate!r}")

        # pitch decomposition is fixed: //12 and %12
        if not bool(pitch_decompose):
            raise ValueError("pitch_decompose is fixed to True (pitch is always split into //12 and %12).")
        if int(pitch_decompose_mod) != 12:
            raise ValueError(f"pitch_decompose_mod is fixed to 12, got {pitch_decompose_mod!r}")
        self.pitch_decompose = True
        self.pitch_decompose_mod = 12
        try:
            r_oct, r_pc = (int(pitch_decompose_fractions[0]), int(pitch_decompose_fractions[1]))
        except Exception as e:
            raise ValueError(
                f"pitch_decompose_fractions must be a tuple[int,int], got {pitch_decompose_fractions!r}"
            ) from e
        if min(r_oct, r_pc) <= 0:
            raise ValueError(f"pitch_decompose_fractions must be positive, got {pitch_decompose_fractions!r}")
        self.pitch_decompose_fractions = (r_oct, r_pc)

        self.pair_layout = str(pair_layout).lower()
        if self.pair_layout not in ('block', 'interleave'):
            raise ValueError(f"pair_layout must be 'block' or 'interleave', got {pair_layout!r}")

        # axis enables (kept as Python bools: no tensor overhead)
        self.enable_index = bool(enable_index)
        self.enable_pos = bool(enable_pos)
        self.enable_pitch = bool(enable_pitch)  # controls both pitch axes
        self.enable_ivl_same = bool(enable_ivl_same)
        self.enable_ivl_prev = bool(enable_ivl_prev)

        # split head_dim/2 pairs across axes
        total_pairs = self.head_dim // 2
        self._init_pair_allocation(total_pairs, axial_fractions)

        if sum(self.mrope_section) != total_pairs:
            raise RuntimeError('Internal error: mrope_section does not sum to total_pairs')

        # Validate interleaved fixed-stride feasibility (enabled axes only)
        if self.pair_layout == 'interleave':
            num_axes = len(self.mrope_section)
            for axis_id, sec in enumerate(self.mrope_section):
                if axis_id == 0 or sec <= 0:
                    continue
                if axis_id == self.AXIS_POS and (not self.enable_pos):
                    continue
                if axis_id in (self.AXIS_PITCH_OCT, self.AXIS_PITCH_PC) and (not self.enable_pitch):
                    continue
                if axis_id == self.AXIS_IVL_SAME and (not self.enable_ivl_same):
                    continue
                if axis_id == self.AXIS_IVL_PREV and (not self.enable_ivl_prev):
                    continue
                max_idx = axis_id + num_axes * (sec - 1)
                if max_idx >= total_pairs:
                    raise ValueError(
                        "pair_layout='interleave' cannot represent this allocation with fixed stride: "
                        f"axis={axis_id} sec={sec} total_pairs={total_pairs} -> max_idx={max_idx}. "
                        "Reduce axial_fractions for non-base axes or use pair_layout='block'."
                    )

        self.reset()

    def _init_pair_allocation(self, total_pairs: int, axial_fractions: tuple[int, ...]) -> None:
        """Initialize mrope_section (pairs per axis) with backward-compatible input formats."""
        ratios = tuple(int(r) for r in axial_fractions)
        r_oct, r_pc = self.pitch_decompose_fractions
        scale_pitch = r_oct + r_pc

        if len(ratios) == 6:
            # (index, pos, pitch_oct, pitch_pc, interval_same, interval_prev)
            if any(r <= 0 for r in ratios):
                raise ValueError(f"axial_fractions must be positive, got {axial_fractions!r}")
            ratios6 = ratios
        elif len(ratios) == 5:
            # (index, pos, pitch_total, interval_same, interval_prev)
            r_index, r_pos, r_pitch, r_is, r_ip = ratios
            if min(r_index, r_pos, r_pitch, r_is, r_ip) <= 0:
                raise ValueError(f"axial_fractions must be positive, got {axial_fractions!r}")
            # scale all non-pitch weights by (r_oct+r_pc) and split pitch_total by (r_oct, r_pc)
            ratios6 = (
                r_index * scale_pitch,
                r_pos * scale_pitch,
                r_pitch * r_oct,
                r_pitch * r_pc,
                r_is * scale_pitch,
                r_ip * scale_pitch,
            )
        elif len(ratios) == 4:
            # legacy: 旧4軸 (index, bar, pos, pitch_total)
            # bar 軸は削除したため、その比率を intervals_total として再利用し、(same, prev) に等分する
            r_index, r_bar, r_pos, r_pitch = ratios
            if min(r_index, r_bar, r_pos, r_pitch) <= 0:
                raise ValueError(f"axial_fractions must be positive, got {axial_fractions!r}")
            # 2 分割（intervals）と (r_oct+r_pc) 分割（pitch）を同時に行うので、全体を 2*scale_pitch で揃える
            scale_all = scale_pitch * 2
            ratios6 = (
                r_index * scale_all,
                r_pos * scale_all,
                r_pitch * r_oct * 2,
                r_pitch * r_pc * 2,
                r_bar * scale_pitch,
                r_bar * scale_pitch,
            )
        else:
            raise ValueError(
                "axial_fractions must have length 6 (index,pos,pitch_oct,pitch_pc,ivl_same,ivl_prev), "
                "or length 5 (index,pos,pitch_total,ivl_same,ivl_prev), "
                "or legacy length 4 (index,bar,pos,pitch_total). "
                f"Got {axial_fractions!r}."
            )

        alloc = self._allocate_pairs(total_pairs, ratios6)
        (
            self.pairs_index,
            self.pairs_pos,
            self.pairs_pitch_oct,
            self.pairs_pitch_pc,
            self.pairs_ivl_same,
            self.pairs_ivl_prev,
        ) = (int(x) for x in alloc)

        # convenience (for compatibility/debug)
        self.pairs_pitch = int(self.pairs_pitch_oct + self.pairs_pitch_pc)

        self.mrope_section = (
            int(self.pairs_index),
            int(self.pairs_pos),
            int(self.pairs_pitch_oct),
            int(self.pairs_pitch_pc),
            int(self.pairs_ivl_same),
            int(self.pairs_ivl_prev),
        )

    @staticmethod
    def _allocate_pairs(total_pairs: int, ratios: tuple[int, ...]) -> tuple[int, ...]:
        """Allocate `total_pairs` into N buckets by integer ratios (deterministic)."""
        rs = [int(r) for r in ratios]
        if any(r <= 0 for r in rs):
            raise ValueError(f"ratios must be positive, got {ratios!r}")
        denom = sum(rs)
        scaled = [total_pairs * r for r in rs]
        base = [s // denom for s in scaled]
        frac = [s % denom for s in scaled]
        rem = total_pairs - sum(base)
        order = sorted(range(len(rs)), key=lambda i: (-frac[i], i))
        for i in order[:rem]:
            base[i] += 1
        return tuple(int(x) for x in base)

    def _axis_cols(self, axis_id: int, axis_pairs: int, *, device: torch.device) -> torch.Tensor:
        """Return column indices for a given axis within full [*, total_pairs] matrices."""
        s = int(axis_pairs)
        if s <= 0:
            return torch.empty((0,), device=device, dtype=torch.int64)

        if self.pair_layout == 'block':
            start = int(sum(int(x) for x in self.mrope_section[:axis_id]))
            return torch.arange(start, start + s, device=device, dtype=torch.int64)
        else:
            stride = len(self.mrope_section)
            return torch.arange(axis_id, axis_id + stride * s, stride, device=device, dtype=torch.int64)

    def _build_pitch_cache(
        self,
        inv_freq: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build pitch cache indexed by pitch_idx (already includes pitch_idx_offset).

        Returns:
            cos, sin: [pitch_cache_len, total_pairs] on `device` with `dtype`.

        実装方針:
          * 参照 index は従来どおり pitch_idx（1 回の embedding lookup）
          * ただし列（rotary pair）を 2 つの pitch 軸セクションに分け、
              - pitch_oct セクションの列は octave_id（pitch//12 + offset）
              - pitch_pc  セクションの列は pc_id（pitch%12 + offset）
            の回転に差し替える（forward の計算増なし）
        """
        inv = inv_freq.to(device=device, dtype=torch.float32)

        t_pitch = torch.arange(int(self.pitch_cache_len), dtype=torch.float32, device=device)
        theta = torch.outer(t_pitch, inv)
        cos = theta.cos()
        sin = theta.sin()

        off = int(self.pitch_idx_offset)
        if off <= 0:
            # non pitch を 0 で分離できないため、分割も行わない（安全側）
            return cos.to(dtype=dtype), sin.to(dtype=dtype)

        # pitch_idx -> raw pitch (>=0) for valid rows
        idx = torch.arange(int(self.pitch_cache_len), device=device, dtype=torch.int32)
        valid = idx >= off
        raw = idx - off
        raw = torch.where(valid, raw, torch.zeros_like(raw))

        mod = 12
        octave = raw // mod
        pc = raw - octave * mod  # 0..11

        octave_id = torch.where(valid, octave + off, torch.zeros_like(raw))
        pc_id = torch.where(valid, pc + off, torch.zeros_like(raw))

        # overwrite only the columns used by each pitch axis
        if int(self.pairs_pitch_oct) > 0:
            cols_oct = self._axis_cols(self.AXIS_PITCH_OCT, int(self.pairs_pitch_oct), device=device)
            inv_oct = inv.index_select(0, cols_oct)
            theta_oct = octave_id.to(torch.float32).unsqueeze(1) * inv_oct.unsqueeze(0)
            cos_oct = theta_oct.cos()
            sin_oct = theta_oct.sin()
            cos.index_copy_(1, cols_oct, cos_oct)
            sin.index_copy_(1, cols_oct, sin_oct)

        if int(self.pairs_pitch_pc) > 0:
            cols_pc = self._axis_cols(self.AXIS_PITCH_PC, int(self.pairs_pitch_pc), device=device)
            inv_pc = inv.index_select(0, cols_pc)
            theta_pc = pc_id.to(torch.float32).unsqueeze(1) * inv_pc.unsqueeze(0)
            cos_pc = theta_pc.cos()
            sin_pc = theta_pc.sin()
            cos.index_copy_(1, cols_pc, cos_pc)
            sin.index_copy_(1, cols_pc, sin_pc)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def set_axes(
        self,
        *,
        index: bool | None = None,
        pos: bool | None = None,
        pitch: bool | None = None,
        ivl_same: bool | None = None,
        ivl_prev: bool | None = None,
    ) -> None:
        """軸の有効/無効を切り替える（必要なら外部から呼ぶ）."""
        if index is not None:
            self.enable_index = bool(index)
        if pos is not None:
            self.enable_pos = bool(pos)
        if pitch is not None:
            self.enable_pitch = bool(pitch)
        if ivl_same is not None:
            self.enable_ivl_same = bool(ivl_same)
        if ivl_prev is not None:
            self.enable_ivl_prev = bool(ivl_prev)

    def _resolve_device(self) -> torch.device:
        dev = globals().get("device", None)
        if dev is not None:
            return dev
        for buf in self.buffers():
            return buf.device
        return torch.device("cpu")

    def reset(self):
        # base inv_freq (full RoPE; no half-truncate)
        dev = self._resolve_device()
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=dev) / self.head_dim)
        )
        # inv_freq participates in YaRN updates and must be checkpointed for exact resume.
        self.inv_freq = nn.Buffer(inv_freq, persistent=True)

        total_pairs = self.head_dim // 2

        # index: [0..T)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=dev)
        theta = torch.outer(t, inv_freq)
        self._cos_index = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=True)
        self._sin_index = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=True)
        assert self._cos_index.size(1) == total_pairs

        # pos: small categorical indices
        t_pos = torch.arange(self.pos_cache_len, dtype=torch.float32, device=dev)
        theta_pos = torch.outer(t_pos, inv_freq)
        self._cos_pos = nn.Buffer(theta_pos.cos().to(torch.bfloat16), persistent=True)
        self._sin_pos = nn.Buffer(theta_pos.sin().to(torch.bfloat16), persistent=True)

        # pitch: categorical indices with fixed split across pitch_oct/pitch_pc axis sections
        cos_pitch, sin_pitch = self._build_pitch_cache(inv_freq, device=dev, dtype=torch.bfloat16)
        self._cos_pitch = nn.Buffer(cos_pitch, persistent=True)
        self._sin_pitch = nn.Buffer(sin_pitch, persistent=True)

        # interval_same: categorical indices (0 is non pitch)
        t_is = torch.arange(self.ivl_same_cache_len, dtype=torch.float32, device=dev)
        theta_is = torch.outer(t_is, inv_freq)
        self._cos_ivl_same = nn.Buffer(theta_is.cos().to(torch.bfloat16), persistent=True)
        self._sin_ivl_same = nn.Buffer(theta_is.sin().to(torch.bfloat16), persistent=True)

        # interval_prev: categorical indices (0 is non pitch / "no prev skyline")
        t_ip = torch.arange(self.ivl_prev_cache_len, dtype=torch.float32, device=dev)
        theta_ip = torch.outer(t_ip, inv_freq)
        self._cos_ivl_prev = nn.Buffer(theta_ip.cos().to(torch.bfloat16), persistent=True)
        self._sin_ivl_prev = nn.Buffer(theta_ip.sin().to(torch.bfloat16), persistent=True)

        # (T,) 用の 0..max_seq_len-1 インデックス（forward で arange を作らない）
        self._idx_cache = nn.Buffer(
            torch.arange(self.max_seq_len, dtype=torch.int32, device=dev), persistent=False
        )

        # attn scale
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        # YaRN frequency scaling（inv_freq を更新し、キャッシュも更新）
        rotations = args.block_size * old_window * self.inv_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.inv_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)

        dev = self.inv_freq.device

        # rebuild index cache
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=dev)
        theta = torch.outer(t, self.inv_freq)
        self._cos_index.copy_(theta.cos().to(self._cos_index.dtype))
        self._sin_index.copy_(theta.sin().to(self._sin_index.dtype))

        # rebuild pos cache
        t_pos = torch.arange(self.pos_cache_len, dtype=torch.float32, device=dev)
        theta_pos = torch.outer(t_pos, self.inv_freq)
        self._cos_pos.copy_(theta_pos.cos().to(self._cos_pos.dtype))
        self._sin_pos.copy_(theta_pos.sin().to(self._sin_pos.dtype))

        # rebuild pitch cache (fixed split)
        cos_pitch, sin_pitch = self._build_pitch_cache(self.inv_freq, device=dev, dtype=self._cos_pitch.dtype)
        self._cos_pitch.copy_(cos_pitch)
        self._sin_pitch.copy_(sin_pitch)

        # rebuild interval caches
        t_is = torch.arange(self.ivl_same_cache_len, dtype=torch.float32, device=dev)
        theta_is = torch.outer(t_is, self.inv_freq)
        self._cos_ivl_same.copy_(theta_is.cos().to(self._cos_ivl_same.dtype))
        self._sin_ivl_same.copy_(theta_is.sin().to(self._sin_ivl_same.dtype))

        t_ip = torch.arange(self.ivl_prev_cache_len, dtype=torch.float32, device=dev)
        theta_ip = torch.outer(t_ip, self.inv_freq)
        self._cos_ivl_prev.copy_(theta_ip.cos().to(self._cos_ivl_prev.dtype))
        self._sin_ivl_prev.copy_(theta_ip.sin().to(self._sin_ivl_prev.dtype))

        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        # token_ids: (T,) を想定（本コードは packed-varlen だが B==1 前提）
        if token_ids.ndim == 2:
            if token_ids.size(0) != 1:
                raise ValueError('This codebase expects packed varlen with B==1; provide token_ids as (T,) or (1,T).')
            token_ids = token_ids.squeeze(0)
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be 1D, got shape={tuple(token_ids.shape)}")

        T = token_ids.size(0)
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds rotary cache length {self.max_seq_len}")

        # Which axes do we actually apply?
        use_pos = self.enable_pos and (self.pairs_pos > 0)
        use_pitch_oct = self.enable_pitch and (self.pairs_pitch_oct > 0)
        use_pitch_pc = self.enable_pitch and (self.pairs_pitch_pc > 0)
        use_pitch_any = use_pitch_oct or use_pitch_pc
        use_is = self.enable_ivl_same and (self.pairs_ivl_same > 0)
        use_ip = self.enable_ivl_prev and (self.pairs_ivl_prev > 0)
        need_extra = use_pos or use_pitch_any or use_is or use_ip

        # Always slice the base cache (cheap view); used as base or identity source
        cos_index = self._cos_index[:T]
        sin_index = self._sin_index[:T]

        # Fast path: pure index (or pure identity) RoPE
        if not need_extra:
            if self.enable_index:
                cos = cos_index
                sin = sin_index
            else:
                # position=0 (cos=1, sin=0) expanded => identity rotation
                cos = cos_index[:1].expand(T, -1)
                sin = sin_index[:1].expand(T, -1)
            return cos[None, :, None, :], sin[None, :, None, :]

        tok = token_ids.to(torch.int32) if token_ids.dtype != torch.int32 else token_ids
        idx = self._idx_cache[:T]

        pitch_tok = (tok >= self.pitch_start) & (tok < (self.pitch_start + self.pitch_size))
        pos_tok = (tok >= self.pos_start) & (tok < (self.pos_start + self.pos_size))
        bar_tok = tok.eq(self.bar_id) if self.bar_id is not None else torch.zeros_like(tok, dtype=torch.bool)
        doc_tok = tok.eq(self.doc_id) if self.doc_id is not None else torch.zeros_like(tok, dtype=torch.bool)

        # gather cos/sin per enabled axis (shape: [T, total_pairs])
        cos_pos = sin_pos = None
        if use_pos:
            # pos index: forward-fill from pos tokens, reset on bar/doc
            pos_updates = torch.where(
                pos_tok,
                (tok - self.pos_start + self.pos_idx_offset).to(dtype=torch.int32),
                torch.zeros_like(tok, dtype=torch.int32),
            )
            pos_update_mask = pos_tok | bar_tok | doc_tok
            pos_idx = forward_fill_from_updates(pos_updates, pos_update_mask, idx)
            pos_idx = pos_idx.clamp(0, self.pos_cache_len - 1)
            cos_pos = F.embedding(pos_idx, self._cos_pos)
            sin_pos = F.embedding(pos_idx, self._sin_pos)

        cos_pitch = sin_pitch = None
        if use_pitch_any:
            # pitch index: forward-fill from pitch tokens, reset optionally
            pitch_updates = torch.where(
                pitch_tok,
                (tok - self.pitch_start + self.pitch_idx_offset).to(dtype=torch.int32),
                torch.zeros_like(tok, dtype=torch.int32),
            )
            if self.carry_pitch:
                pitch_update_mask = pitch_tok | doc_tok
                if self.reset_pitch_on_pos:
                    pitch_update_mask = pitch_update_mask | pos_tok
                if self.reset_pitch_on_bar:
                    pitch_update_mask = pitch_update_mask | bar_tok
                pitch_idx = forward_fill_from_updates(pitch_updates, pitch_update_mask, idx)
            else:
                pitch_idx = pitch_updates

            if self.pitch_gate == 'pitch_tok_only':
                pitch_idx = pitch_idx.masked_fill(~pitch_tok, 0)

            pitch_idx = pitch_idx.clamp(0, self.pitch_cache_len - 1)
            cos_pitch = F.embedding(pitch_idx, self._cos_pitch)
            sin_pitch = F.embedding(pitch_idx, self._sin_pitch)

        # pitch absolute value in [0..pitch_size-1] for pitch tokens (undefined elsewhere)
        pitch_abs = (tok - self.pitch_start).to(dtype=torch.int32)

        cos_is = sin_is = None
        cos_ip = sin_ip = None
        if use_is or use_ip:
            # skyline token: first pitch immediately following a POS token
            prev_is_pos = torch.cat([pos_tok.new_zeros((1,)), pos_tok[:-1]], dim=0)
            skyline_tok = pitch_tok & prev_is_pos

            # interval_same: diff to skyline at the same POS group
            if use_is:
                skyline_updates = torch.where(skyline_tok, pitch_abs, torch.zeros_like(pitch_abs))
                skyline_update_mask = skyline_tok | pos_tok | bar_tok | doc_tok
                skyline_pitch = forward_fill_from_updates(skyline_updates, skyline_update_mask, idx)

                diff_same = (pitch_abs - skyline_pitch).to(torch.int32)

                ivl_same_idx = torch.where(
                    pitch_tok,
                    (diff_same - int(self.ivl_same_min) + int(self.ivl_idx_offset)).to(torch.int32),
                    torch.zeros_like(pitch_abs),
                )
                ivl_same_idx = ivl_same_idx.clamp(0, self.ivl_same_cache_len - 1)
                cos_is = F.embedding(ivl_same_idx, self._cos_ivl_same)
                sin_is = F.embedding(ivl_same_idx, self._sin_ivl_same)

            # interval_prev: diff to previous skyline (piece start => non pitch)
            if use_ip:
                skyline_id_updates = torch.where(skyline_tok, pitch_abs + 1, torch.zeros_like(pitch_abs))
                last_skyline_id = forward_fill_from_updates(
                    skyline_id_updates, skyline_tok | doc_tok, idx
                )

                prev_seg_updates = torch.where(pos_tok, last_skyline_id, torch.zeros_like(pitch_abs))
                prev_seg_id = forward_fill_from_updates(
                    prev_seg_updates, pos_tok | bar_tok | doc_tok, idx
                )

                have_prev = prev_seg_id > 0
                prev_pitch_abs = (prev_seg_id - 1).to(torch.int32)  # -1 if have_prev==False
                diff_prev = (pitch_abs - prev_pitch_abs).to(torch.int32)

                ivl_prev_idx = torch.where(
                    pitch_tok & have_prev,
                    (diff_prev - int(self.ivl_prev_min) + int(self.ivl_idx_offset)).to(torch.int32),
                    torch.zeros_like(pitch_abs),
                )
                ivl_prev_idx = ivl_prev_idx.clamp(0, self.ivl_prev_cache_len - 1)
                cos_ip = F.embedding(ivl_prev_idx, self._cos_ivl_prev)
                sin_ip = F.embedding(ivl_prev_idx, self._sin_ivl_prev)

        # base: index RoPE or identity
        if self.enable_index:
            cos_base = cos_index
            sin_base = sin_index
        else:
            cos_base = cos_index[:1].expand(T, -1)
            sin_base = sin_index[:1].expand(T, -1)

        # Merge: clone once, then overwrite enabled axes only (no extra elementwise masking)
        cos = cos_base.clone()
        sin = sin_base.clone()

        if self.pair_layout == 'block':
            s0, s1, s2, s3, s4, s5 = self.mrope_section
            o1 = s0
            o2 = s0 + s1
            o3 = o2 + s2
            o4 = o3 + s3
            o5 = o4 + s4
            if use_pos:
                cos[:, o1:o2] = cos_pos[:, o1:o2]
                sin[:, o1:o2] = sin_pos[:, o1:o2]
            if use_pitch_oct:
                cos[:, o2:o3] = cos_pitch[:, o2:o3]
                sin[:, o2:o3] = sin_pitch[:, o2:o3]
            if use_pitch_pc:
                cos[:, o3:o4] = cos_pitch[:, o3:o4]
                sin[:, o3:o4] = sin_pitch[:, o3:o4]
            if use_is:
                cos[:, o4:o5] = cos_is[:, o4:o5]
                sin[:, o4:o5] = sin_is[:, o4:o5]
            if use_ip:
                cos[:, o5:] = cos_ip[:, o5:]
                sin[:, o5:] = sin_ip[:, o5:]
        else:
            # interleaved fixed-stride overwrite
            stride = len(self.mrope_section)
            if use_pos:
                end = int(self.pairs_pos) * stride
                cos[:, 1:end:stride] = cos_pos[:, 1:end:stride]
                sin[:, 1:end:stride] = sin_pos[:, 1:end:stride]
            if use_pitch_oct:
                end = int(self.pairs_pitch_oct) * stride
                cos[:, 2:end:stride] = cos_pitch[:, 2:end:stride]
                sin[:, 2:end:stride] = sin_pitch[:, 2:end:stride]
            if use_pitch_pc:
                end = int(self.pairs_pitch_pc) * stride
                cos[:, 3:end:stride] = cos_pitch[:, 3:end:stride]
                sin[:, 3:end:stride] = sin_pitch[:, 3:end:stride]
            if use_is:
                end = int(self.pairs_ivl_same) * stride
                cos[:, 4:end:stride] = cos_is[:, 4:end:stride]
                sin[:, 4:end:stride] = sin_is[:, 4:end:stride]
            if use_ip:
                end = int(self.pairs_ivl_prev) * stride
                cos[:, 5:end:stride] = cos_ip[:, 5:end:stride]
                sin[:, 5:end:stride] = sin_ip[:, 5:end:stride]

        # shape to [1, T, 1, D/2] for apply_rotary_emb()
        return cos[None, :, None, :], sin[None, :, None, :]
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@dataclass
class AttnArgs:
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    block_mask: BlockMask


def _dense_to_ordered_blockmask(dense_blockmask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
    dense_i32 = dense_blockmask.to(dtype=torch.int32)
    indices = dense_i32.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()


def _build_varlen_flex_block_masks(
    seqlens: torch.Tensor,
    total_len: int,
    window_sizes_tokens: set[int],
    *,
    block_size: int,
    device: torch.device,
) -> dict[int, BlockMask]:
    total_len = int(total_len)
    assert total_len > 0
    assert seqlens.ndim == 1 and seqlens.numel() >= 2

    seqlens_i64 = seqlens.to(device=device, dtype=torch.int64)
    doc_lengths = (seqlens_i64[1:] - seqlens_i64[:-1]).clamp_min_(0)
    doc_ids = torch.arange(doc_lengths.numel(), device=device, dtype=torch.int32)
    docs = torch.repeat_interleave(doc_ids, doc_lengths)
    if docs.numel() != total_len:
        raise RuntimeError(
            f"seqlens and total_len mismatch: docs={docs.numel()} total_len={total_len}"
        )

    num_blocks = (total_len + block_size - 1) // block_size
    block_idx = torch.arange(num_blocks, device=device, dtype=torch.int32)
    block_starts = block_idx.to(torch.int64) * int(block_size)
    block_ends = torch.clamp(block_starts + int(block_size) - 1, max=total_len - 1)
    doc_ends = seqlens_i64[1:]

    docs_low = torch.bucketize(block_starts, doc_ends, right=False).to(torch.int32)
    docs_high = torch.bucketize(block_ends, doc_ends, right=False).to(torch.int32)

    delta = block_idx[:, None] - block_idx[None, :]
    causal_blockmask_any = delta >= 0
    causal_blockmask_all = delta > 0
    document_blockmask_any = (
        (docs_low[:, None] <= docs_high[None, :])
        & (docs_high[:, None] >= docs_low[None, :])
    )
    document_blockmask_all = (
        (docs_low[:, None] == docs_high[None, :])
        & (docs_high[:, None] == docs_low[None, :])
    )

    aligned_to_block = (total_len % int(block_size) == 0)

    def make_mask_mod(window_tokens: int):
        if aligned_to_block:
            # Fast path (train/val): compile-friendly mask_mod equivalent to flex.py semantics.
            def document_causal_local_window(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx]) & ((q_idx - kv_idx) < window_tokens)
            return document_causal_local_window

        # Safe path (ragged lengths): guard tail indices for non block-aligned prefill.
        def document_causal_local_window(b, h, q_idx, kv_idx):
            valid = (q_idx < total_len) & (kv_idx < total_len)
            q_safe = q_idx.clamp_max(total_len - 1)
            kv_safe = kv_idx.clamp_max(total_len - 1)
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_safe] == docs[kv_safe]
            return valid & causal_mask & document_mask & ((q_idx - kv_idx) < window_tokens)
        return document_causal_local_window

    out: dict[int, BlockMask] = {}
    for window_tokens in sorted(int(x) for x in window_sizes_tokens):
        if window_tokens <= 0:
            raise ValueError(f"window_tokens must be > 0, got {window_tokens}")
        window_blocks = (window_tokens + block_size - 1) // block_size
        window_blockmask_any = delta <= window_blocks
        window_blockmask_all = delta <= max(window_blocks - 1, 0)
        blockmask_any = causal_blockmask_any & document_blockmask_any & window_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all & window_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = _dense_to_ordered_blockmask(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = _dense_to_ordered_blockmask(blockmask_all)
        out[window_tokens] = BlockMask.from_kv_blocks(
            partial_kv_num_blocks,
            partial_kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=make_mask_mod(window_tokens),
        )
    return out


def _build_causal_window_block_mask(
    *,
    q_len: int,
    kv_len: int,
    q_start: int,
    window_tokens: int,
    block_size: int,
    device: torch.device,
) -> BlockMask:
    q_len = int(q_len)
    kv_len = int(kv_len)
    q_start = int(q_start)
    window_tokens = int(window_tokens)
    if q_len <= 0 or kv_len <= 0:
        raise ValueError(f"Invalid q/kv lengths: q_len={q_len}, kv_len={kv_len}")
    if window_tokens <= 0:
        raise ValueError(f"window_tokens must be > 0, got {window_tokens}")

    def causal_local_window(b, h, q_idx, kv_idx):
        q_global = q_idx + q_start
        return (q_global >= kv_idx) & ((q_global - kv_idx) < window_tokens)

    return create_block_mask(
        causal_local_window,
        B=1,
        H=1,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device.type,
        BLOCK_SIZE=block_size,
        _compile=False,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int, num_layers: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.n_kv_head = num_kv_heads
        self.head_dim = head_dim
        self.dim = dim
        assert self.num_heads * self.head_dim == self.dim
        assert self.n_kv_head <= self.num_heads and self.num_heads % self.n_kv_head == 0

        self.c_q = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, num_layers) else None


    def forward(self, x: Tensor, ve: Tensor | None, attn_args: AttnArgs):
        B, T, _ = x.size()
        assert B == 1, "varlen sequences requires B == 1"

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos = attn_args.cos[:, :T]
        sin = attn_args.sin[:, :T]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = _flex_attention_call(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=attn_args.block_mask,
            scale=attn_args.attn_scale,
            enable_gqa=(self.num_heads != self.n_kv_head),
        ).transpose(1, 2)
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int, num_layers: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, num_kv_heads, num_layers, layer_idx)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, attn_args: AttnArgs):
        x = x + self.attn(norm(x), ve, attn_args)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


@dataclass
class ForwardScheduleConfig:
    ws_short: int
    ws_long: int


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        assert self.num_kv_heads <= self.num_heads and self.num_heads % self.num_kv_heads == 0

        self.vocab_size = vocab_size
        self.padded_vocab_size = next_multiple_of_n(vocab_size, n=64)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(self.padded_vocab_size, model_dim),
            "h": nn.ModuleList([Block(model_dim, head_dim, num_heads, num_kv_heads, num_layers, i) for i in range(num_layers)]),
        })
        self.lm_head = nn.Linear(model_dim, self.padded_vocab_size, bias=False)

        # Per-layer scalars
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))

        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        kv_dim = num_kv_heads * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(self.padded_vocab_size, kv_dim)
            for i in range(num_layers) if has_ve(i, num_layers)
        })

        # YaRN scaling on top of nanochat RoPE
        self.yarn = Yarn(head_dim, max_seq_len)

        # nanochat-style window pattern (tiled across layers, final layer always long)
        self.window_pattern = args.window_pattern.upper()
        if not self.window_pattern or any(c not in "SL" for c in self.window_pattern):
            raise ValueError(f"Invalid window_pattern: {args.window_pattern}. Use only S and L.")

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        s = (3 ** 0.5) * (self.model_dim ** -0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.0)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

    def setup_optimizers(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.8, 0.95),
        scalar_lr: float = 0.5,
    ):
        model_dim = self.model_dim
        ddp = dist.is_initialized()
        # Separate out parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params)
        )
        # AdamW optimizer for embedding, lm_head, and per-layer scalars
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Muon optimizer for matrix parameters
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def _get_bm_sizes(self, ws_short: int, ws_long: int):
        # Cache per (ws_short, ws_long) to avoid per-token Python overhead in inference.
        cache = getattr(self, "_bm_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_bm_cache", cache)
        key = (int(ws_short), int(ws_long))
        cached = cache.get(key)
        if cached is not None:
            return cached

        short_bm = int(ws_short) * args.block_size
        long_bm = int(ws_long) * args.block_size
        pattern = self.window_pattern
        window_sizes = []
        for layer_idx in range(self.num_layers):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(long_bm if char == "L" else short_bm)
        window_sizes[-1] = long_bm
        cache[key] = window_sizes
        return window_sizes

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        schedule_cfg: ForwardScheduleConfig,
        max_seqlen: int | None = None,
    ):
        assert input_seq.ndim == 1

        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

        B = 1
        T = input_seq.size(0)
        assert T <= self.yarn.max_seq_len, "sequence length exceeds rotary cache"

        x = self.transformer.wte(input_seq)
        x = norm(x)[None]
        x0 = x

        bm_sizes = self._get_bm_sizes(ws_short, ws_long)
        block_masks = _build_varlen_flex_block_masks(
            seqlens=seqlens,
            total_len=T,
            window_sizes_tokens=set(bm_sizes),
            block_size=int(args.block_size),
            device=input_seq.device,
        )

        # YaRN + 4-axis MRoPE: generate cos/sin once per sequence and share across layers
        cos, sin = self.yarn(input_seq)

        # Cast per-layer scalars once per forward to avoid tiny per-layer allocations.
        a_vec = self.resid_lambdas.to(dtype=x.dtype)
        b_vec = self.x0_lambdas.to(dtype=x.dtype)

        for i, block in enumerate(self.transformer.h):
            a = a_vec[i]
            b = b_vec[i]
            x = a * x + b * x0
            ve = self.value_embeds[str(i)](input_seq) if str(i) in self.value_embeds else None
            attn_args = AttnArgs(
                cos=cos,
                sin=sin,
                attn_scale=self.yarn.attn_scale,
                block_mask=block_masks[bm_sizes[i]],
            )
            x = block(x, ve, attn_args)
        x = norm(x)

        softcap = 15.0

        if not self.training:
            x_flat = x.flatten(end_dim=1)
            x_chunks = x_flat.chunk(4)
            t_chunks = target_seq.chunk(4)
            total = target_seq.numel()
            loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            for x_chunk, t_chunk in zip(x_chunks, t_chunks):
                if t_chunk.numel() == 0:
                    continue
                logits = F.linear(x_chunk, self.lm_head.weight).float()
                logits = logits[:, :self.vocab_size]
                logits = softcap * torch.tanh(logits / softcap)
                loss_sum += F.cross_entropy(logits, t_chunk, reduction="sum")
            return loss_sum / max(total, 1)

        logits = self.lm_head(x)
        logits = logits[..., :self.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        logits_for_loss = logits
        loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="mean")
        return loss
# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def _concat_slices(tokens: torch.Tensor, start_idxs: torch.Tensor, end_idxs: torch.Tensor) -> torch.Tensor:
    """Concatenate token slices [start:end) without repeated torch.cat allocations."""
    lengths = (end_idxs - start_idxs).to(dtype=torch.int64)
    total = int(lengths.sum().item())
    if total <= 0:
        return torch.empty((0,), dtype=tokens.dtype)
    out = torch.empty(total, dtype=tokens.dtype)
    offset = 0
    for s, e in zip(start_idxs.tolist(), end_idxs.tolist()):
        seg = tokens[s:e]
        n = int(seg.numel())
        if n:
            out[offset:offset + n] = seg
            offset += n
    return out

def _make_cu_seqlens(cum_lengths: torch.Tensor, total_len: int) -> torch.Tensor:
    """Build cu_seqlens starting at 0 and ending at total_len."""
    total_len = int(total_len)
    if cum_lengths.numel() == 0:
        return torch.tensor([0, total_len], dtype=torch.int32)
    if int(cum_lengths[0].item()) == 0:
        cum_lengths = cum_lengths[1:]
        if cum_lengths.numel() == 0:
            return torch.tensor([0, total_len], dtype=torch.int32)
    last = int(cum_lengths[-1].item())
    if last != total_len:
        cum_lengths = torch.cat([cum_lengths, cum_lengths.new_tensor([total_len])], dim=0)
    out = torch.empty((cum_lengths.numel() + 1,), dtype=torch.int32)
    out[0] = 0
    out[1:] = cum_lengths.to(dtype=torch.int32)
    return out

BOS_ID = REMI96.bos_id

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter==5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter+=1
        return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = cycle(files) if len(files) == 1 else iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = _concat_slices(tokens, start_idxs, end_idxs)
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            doc_lengths = end_idxs - start_idxs
            cum_lengths = doc_lengths.cumsum(0)
            max_seqlen = int(doc_lengths.max().item()) if doc_lengths.numel() else int(max_seq_len)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            max_seqlen = int(max_seq_len) if int(max_seq_len) > 0 else int(num_tokens_local)
            pos += num_tokens


        _cum_lengths = _make_cu_seqlens(cum_lengths, num_tokens_local)

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            int(max_seqlen),
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len


class ResumableDistributedDataLoader:
    """Resumable variant of `distributed_data_generator`.

    * API-compatible with the existing training loop: `.send(new_params)` returns
      `(inputs, targets, cum_seqlens, max_seqlen)`.
    * Captures enough state to resume deterministically at step boundaries:
      - current shard index + BOSFinder position
      - next shard prefetch indices (to preserve shard order)
      - current (num_tokens, max_seq_len, grad_accum_steps)
    """

    def __init__(
        self,
        filename_pattern: str,
        num_tokens: int,
        max_seq_len: int,
        *,
        grad_accum_steps: int = 1,
        align_to_bos: bool = True,
    ):
        self.filename_pattern = filename_pattern
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.align_to_bos = bool(align_to_bos)
        self.max_seq_len = int(max_seq_len)

        assert num_tokens % (self.world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
        self.num_tokens_total = int(num_tokens)
        self.grad_accum_steps = int(grad_accum_steps)
        # internal: tokens per micro-batch (per grad-accum step), matching distributed_data_generator
        self._num_tokens = self.num_tokens_total // self.grad_accum_steps

        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
        self._cycle = (len(self.files) == 1)

        # shard pointers
        self.cur_file_idx = 0
        self.prefetch_file_idx = None
        self.file_ptr_idx = None

        # shard state
        self.tokens: Tensor | None = None
        self.finder: BOSFinder | None = None
        self.pos: int = 0  # only used when align_to_bos=False

        # async prefetch
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_ready = threading.Event()
        self._prefetch_data = None  # (file_idx, tokens, finder)

        # initial load + prefetch
        self._load_shard(self.cur_file_idx, quickload=(self.align_to_bos is True))
        if self.align_to_bos:
            self._init_prefetch_indices()
            self._start_prefetch()

    # --------------------- internal helpers ---------------------

    def _init_prefetch_indices(self) -> None:
        if self._cycle:
            self.prefetch_file_idx = 0
            self.file_ptr_idx = 0
            return
        # next file after current
        nxt = self.cur_file_idx + 1
        if nxt >= len(self.files):
            self.prefetch_file_idx = None
            self.file_ptr_idx = None
        else:
            self.prefetch_file_idx = nxt
            self.file_ptr_idx = nxt + 1

    def _advance_file_ptr(self, idx: int | None) -> int | None:
        if idx is None:
            return None
        if self._cycle:
            return 0
        if idx >= len(self.files):
            return None
        return idx

    def _load_shard(self, file_idx: int, *, quickload: bool) -> None:
        file = self.files[file_idx]
        tokens = _load_data_shard(file)
        self.tokens = tokens
        if self.align_to_bos:
            # quickload=True only for the *current* shard (matches distributed_data_generator)
            self.finder = BOSFinder(tokens, world_size=self.world_size, quickload=quickload)
        else:
            self.pos = 0
            self.finder = None

    def _prefetch_worker(self, file_idx: int | None) -> None:
        try:
            if file_idx is None:
                self._prefetch_data = None
            else:
                tokens = _load_data_shard(self.files[file_idx])
                finder = BOSFinder(tokens, self.world_size)
                self._prefetch_data = (file_idx, tokens, finder)
        finally:
            self._prefetch_ready.set()

    def _start_prefetch(self) -> None:
        # nothing to prefetch
        if self.prefetch_file_idx is None:
            self._prefetch_thread = None
            self._prefetch_data = None
            self._prefetch_ready.set()
            return

        # start async prefetch for the *next* shard
        self._prefetch_ready.clear()
        file_idx = int(self.prefetch_file_idx)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(file_idx,), daemon=True
        )
        self._prefetch_thread.start()

        # NOTE: We intentionally do not mutate file_ptr_idx here.
        # In this class, `prefetch_file_idx` is "the shard currently being prefetched",
        # and `file_ptr_idx` is "the shard to prefetch after the current prefetch is consumed".
        # This is already the post-prefetch position, matching the original iterator semantics.

    def _consume_prefetch(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_ready.wait()
            self._prefetch_thread.join()
        data = self._prefetch_data
        if data is None:
            raise StopIteration("Dataset exhausted (no prefetched shard available).")
        file_idx, tokens, finder = data

        # swap current shard
        self.cur_file_idx = int(file_idx)
        self.tokens = tokens
        self.finder = finder

        # set up next prefetch indices
        if self._cycle:
            self.prefetch_file_idx = 0
            self.file_ptr_idx = 0
        else:
            # advance along the file list (no cycling)
            if self.file_ptr_idx is None or self.file_ptr_idx >= len(self.files):
                self.prefetch_file_idx = None
                self.file_ptr_idx = None
            else:
                self.prefetch_file_idx = int(self.file_ptr_idx)
                nxt = int(self.file_ptr_idx) + 1
                self.file_ptr_idx = nxt if nxt < len(self.files) else None

        # kick off prefetch for the next shard
        self._start_prefetch()

    # --------------------- public API ---------------------

    def send(self, new_params):
        """Generator-compatible API.

        Args:
            new_params: None or (new_num_tokens, new_max_seq_len, new_grad_accum_steps)
        """
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (self.world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            self.num_tokens_total = int(new_num_tokens)
            self.grad_accum_steps = int(new_grad_accum_steps)
            self._num_tokens = self.num_tokens_total // self.grad_accum_steps
            self.max_seq_len = int(new_max_seq_len)

        while True:
            num_tokens_local = self._num_tokens // self.world_size
            if self.align_to_bos:
                assert self.finder is not None and self.tokens is not None
                try:
                    seq_starts, seq_ends = self.finder.next_batch(num_tokens_local, self.max_seq_len)
                    start_idxs = torch.tensor(seq_starts[self.rank])
                    end_idxs = torch.tensor(seq_ends[self.rank])
                except StopIteration:
                    # shard exhausted: move to the prefetched shard
                    self._consume_prefetch()
                    continue

                buf = _concat_slices(self.tokens, start_idxs, end_idxs)
                _inputs = buf[:-1]
                _targets = buf[1:]
                end_idxs[-1] -= 1
                doc_lengths = end_idxs - start_idxs
                cum_lengths = doc_lengths.cumsum(0)
                max_seqlen = int(doc_lengths.max().item()) if doc_lengths.numel() else int(self.max_seq_len)

            else:
                assert self.tokens is not None
                if self.pos + self._num_tokens + 1 >= len(self.tokens):
                    # In the original generator this is "should not occur for val data".
                    # Keep behavior: load next shard in order.
                    if self._cycle:
                        self.cur_file_idx = 0
                    else:
                        if self.cur_file_idx + 1 >= len(self.files):
                            raise StopIteration("Dataset exhausted (unaligned).")
                        self.cur_file_idx += 1
                    self._load_shard(self.cur_file_idx, quickload=False)
                    self.pos = 0

                pos_local = self.pos + self.rank * num_tokens_local
                buf = self.tokens[pos_local: pos_local + num_tokens_local + 1]
                _inputs = buf[:-1].view(num_tokens_local, )
                _targets = buf[1:].view(num_tokens_local, )

                cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
                max_seqlen = int(self.max_seq_len) if int(self.max_seq_len) > 0 else int(num_tokens_local)
                self.pos += self._num_tokens

            _cum_lengths = _make_cu_seqlens(cum_lengths, num_tokens_local)

            _inputs = _inputs.to(dtype=torch.int32)
            _targets = _targets.to(dtype=torch.int64)
            _cum_lengths = _cum_lengths.to(dtype=torch.int32)

            return (
                _inputs.to(device="cuda", non_blocking=True),
                _targets.to(device="cuda", non_blocking=True),
                _cum_lengths.to(device="cuda", non_blocking=True),
                int(max_seqlen),
            )

    def state_dict(self) -> dict:
        """Return a lightweight, pickleable state."""
        st = dict(
            version=1,
            filename_pattern=self.filename_pattern,
            align_to_bos=self.align_to_bos,
            num_tokens_total=self.num_tokens_total,
            grad_accum_steps=self.grad_accum_steps,
            max_seq_len=self.max_seq_len,
            cur_file_idx=self.cur_file_idx,
            prefetch_file_idx=self.prefetch_file_idx,
            file_ptr_idx=self.file_ptr_idx,
            # store file paths as an integrity check (and to survive changes in glob ordering)
            cur_file=str(self.files[self.cur_file_idx]),
            prefetch_file=(
                str(self.files[self.prefetch_file_idx])
                if self.prefetch_file_idx is not None and 0 <= self.prefetch_file_idx < len(self.files)
                else None
            ),
            file_ptr_file=(
                str(self.files[self.file_ptr_idx])
                if self.file_ptr_idx is not None and 0 <= self.file_ptr_idx < len(self.files)
                else None
            ),
        )
        if self.align_to_bos:
            assert self.finder is not None
            st.update(
                finder_i=int(self.finder.i),
                finder_batch_iter=int(self.finder.batch_iter),
                finder_quickload=bool(self.finder.quickload),
            )
        else:
            st.update(pos=int(self.pos))
        return st

    def load_state_dict(self, st: dict) -> None:
        """Restore from `state_dict()`."""
        # If a prefetch is currently in-flight from the constructor, join it to avoid
        # races when we overwrite indices and restart prefetching.
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_ready.wait()
            self._prefetch_thread.join()
            self._prefetch_thread = None
            self._prefetch_data = None
        assert int(st.get("version", 0)) == 1, f"Unsupported dataloader state version: {st.get('version')}"
        assert bool(st["align_to_bos"]) == self.align_to_bos, "align_to_bos mismatch"
        assert str(st["filename_pattern"]) == self.filename_pattern, "filename_pattern mismatch"

        self.num_tokens_total = int(st["num_tokens_total"])
        self.grad_accum_steps = int(st["grad_accum_steps"])
        self._num_tokens = self.num_tokens_total // self.grad_accum_steps
        self.max_seq_len = int(st["max_seq_len"])

        # Resolve indices from file paths if available (robust against glob ordering changes).
        def _idx_from_path(p: str | None) -> int | None:
            if p is None:
                return None
            for i, f in enumerate(self.files):
                if str(f) == p:
                    return i
            return None

        cur_idx = _idx_from_path(st.get("cur_file", None))
        self.cur_file_idx = int(cur_idx) if cur_idx is not None else int(st["cur_file_idx"])

        pre_idx = _idx_from_path(st.get("prefetch_file", None))
        if pre_idx is None:
            self.prefetch_file_idx = st.get("prefetch_file_idx", None)
            if self.prefetch_file_idx is not None:
                self.prefetch_file_idx = int(self.prefetch_file_idx)
        else:
            self.prefetch_file_idx = int(pre_idx)

        ptr_idx = _idx_from_path(st.get("file_ptr_file", None))
        if ptr_idx is None:
            self.file_ptr_idx = st.get("file_ptr_idx", None)
            if self.file_ptr_idx is not None:
                self.file_ptr_idx = int(self.file_ptr_idx)
        else:
            self.file_ptr_idx = int(ptr_idx)

        # restore current shard
        if self.align_to_bos:
            batch_iter = int(st.get("finder_batch_iter", 0))
            # If we are beyond the quickload region, we need the full BOS index.
            quickload = bool(st.get("finder_quickload", True)) and (batch_iter < 5)
            self._load_shard(self.cur_file_idx, quickload=quickload)
            assert self.finder is not None
            self.finder.i = int(st["finder_i"])
            self.finder.batch_iter = batch_iter
        else:
            self._load_shard(self.cur_file_idx, quickload=False)
            self.pos = int(st["pos"])

        # restart prefetch thread (we do not attempt to restore in-flight buffers)
        if self.align_to_bos:
            self._start_prefetch()

# -----------------------------------------------------------------------------
# Training Management

def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]

def get_ws(step: int):
    # set short window size to half of long window size
    # Higher ws on "extension" steps
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

# learning rate schedule: nanochat warmup/warmdown
def get_lr(step: int):
    warmup_iters = round(args.warmup_ratio * args.num_iterations)
    warmdown_iters = round(args.warmdown_ratio * args.num_iterations)
    if warmup_iters > 0 and step < warmup_iters:
        return (step + 1) / warmup_iters
    if warmdown_iters == 0 or step <= args.num_iterations - warmdown_iters:
        return 1.0
    progress = (args.num_iterations - step) / warmdown_iters
    return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(step: int, muon_warmup_steps=300, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    frac = min(step / muon_warmup_steps, 1.0)
    return momentum_min + frac * (momentum_max - momentum_min)

def get_weight_decay(step: int, weight_decay_scaled: float):
    # linear decay to zero over the course of training
    progress = step / max(args.num_iterations, 1)
    return weight_decay_scaled * (1 - progress)

class TrainingManager():
    """
    Manages two optimizers (AdamW for embeddings/lm_head/scalars, Muon for matrices).
    Notable Features:
        1. AdamW + Muon split matches nanochat optimizer design
        2. Learning rates follow nanochat warmup/warmdown schedule
        3. Muon momentum warmup and linear weight decay schedule
        4. Embed/lm_head are untied (nanochat-style)

    Manages model architecture, data, and target that changes during training   
    Notable Features:
        1. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        2. YaRN updates to RoPE on window changes
        3. Untied embed and lm_head (nanochat-style)
        4. Token-based batch size schedule via train_bs_schedule
        5. Post training extension of long windows from 13 to 20
    """
    def __init__(self, model):
        self.model = model

        self.reference_batch_size = args.reference_batch_size
        self.weight_decay_scaled = args.weight_decay * (12 / model.num_layers) ** 2
        if model.num_layers != 12:
            print0(f"Scaling weight decay from {args.weight_decay:.6f} to {self.weight_decay_scaled:.6f} for depth {model.num_layers}")

        adam_betas = (args.adam_beta1, args.adam_beta2)
        self.optimizers = model.setup_optimizers(
            unembedding_lr=args.unembedding_lr,
            embedding_lr=args.embedding_lr,
            matrix_lr=args.matrix_lr,
            weight_decay=self.weight_decay_scaled,
            adam_betas=adam_betas,
            scalar_lr=args.scalar_lr,
        )
        self.adamw_opt, self.muon_opt = self.optimizers

        self.reset()

    def apply_final_ws_ext(self):
        new_ws_long = args.ws_validate_post_yarn_ext
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long, new_ws_long)
            self.ws_long = new_ws_long
        self.ws_short = self.ws_long // 2

    def get_forward_args(self):
        return ForwardScheduleConfig(
            ws_short = self.ws_short,
            ws_long = self.ws_long
        )
    
    def get_transition_steps(self):
        transition_steps = [0]
        ws_short, ws_long = get_ws(0)
        for step in range(1, args.num_iterations):
            ws_short, new_ws_long = get_ws(step)
            if new_ws_long != ws_long:
                transition_steps.append(step)
                ws_long = new_ws_long
        return transition_steps

    def advance_schedule(self, step: int):
        self.ws_short, new_ws_long = get_ws(step)
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long, new_ws_long)

        new_batch_size = get_bs(step)
        if new_batch_size != self.batch_size:
            new_grad_accum_steps = compute_grad_accum_steps(new_batch_size)
            batch_ratio = new_batch_size / self.reference_batch_size
            self.batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0
            print0(
                f"Scaling LRs by {self.batch_lr_scale:.4f} for batch size {new_batch_size:,} "
                f"(reference: {self.reference_batch_size:,})"
            )
            self.train_loader_send_args = (new_batch_size, args.train_max_seq_len, new_grad_accum_steps)
            self.grad_accum_steps = new_grad_accum_steps
            self.batch_size = new_batch_size
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long

    def step_optimizers(self, step: int):
        step_lr = get_lr(step)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step, self.weight_decay_scaled)
        for group in self.muon_opt.param_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * self.batch_lr_scale * step_lr
            opt.step()
        self.model.zero_grad(set_to_none=True)

    def reset(self, state=None):
        if state is not None:
            for opt, opt_state in zip(self.optimizers, state):
                opt.load_state_dict(opt_state)

        # muon momentum buffers not in state dict
        self.muon_opt.reset()

        self.ws_short, self.ws_long = get_ws(0)
        self.batch_size = get_bs(0)
        self.grad_accum_steps = compute_grad_accum_steps(self.batch_size)
        batch_ratio = self.batch_size / self.reference_batch_size
        self.batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0
        self.model.yarn.reset()

    def get_state(self):
        return [copy.deepcopy(opt.state_dict()) for opt in self.optimizers]

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "/content/test1/*.bin" # input .bin to train on
    val_files: str = "val.bin" # input .bin to eval validation loss on
    val_tokens: int = 32 * 2048 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    train_bs_schedule: tuple = (32 * 2048,32 * 2048)
    train_bs_extension: int = 32 * 2048
    train_max_seq_len: int = 32 * 2048 # doubled to enable longer window sizes
    val_batch_size: int = 32 * 2048
    device_batch_size_tokens: int = 32 * 2048  # per-rank sequence length (varlen B==1)
    reference_batch_size: int = 2**19
    # optimization
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.3
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    num_scheduled_iterations: int = 5000  # number of steps to complete ws schedule
    num_extension_iterations: int = 500  # number of steps to continue training at final lr and ws
    num_iterations: int = num_scheduled_iterations + num_extension_iterations   
    # nanochat-style LR schedule
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.6
    final_lr_frac: float = 0.0
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 100  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = True
    # checkpointing / resume
    save_every: int = 1000  # (steps) 0 disables periodic checkpoints (keeps legacy behavior)
    checkpoint_dir: str = "/content/1"  # "" -> logs/{run_id}
    resume: bool = False
    resume_step: int = -1  # -1 -> latest
    # attention masking
    block_size: int = 128
    window_pattern: str = "SSSL"
    ws_schedule: tuple = (3, 7, 11, 15, 19, 23, 27)
    ws_final: int = 27 # set final validation ws, used for YaRN extension and short window size
    ws_validate_post_yarn_ext: int = 27 # extend long windows out even further after applying YaRN
    # model (GQA) - 0 means use num_heads (GQA disabled, nanochat default)
    num_kv_heads: int = 0

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)
args.total_batch_size_tokens = args.train_bs_schedule[0]

# -----------------------------------------------------------------------------
# Inference utilities (KV-cache toggle, MIDI <-> token, FlexAttention)
# -----------------------------------------------------------------------------
from typing import Optional, Sequence, Tuple

class KVCache:
    """Per-layer KV cache for FlexAttention kv-cache inference.

    Shapes:
      k_cache, v_cache: [num_layers, B, max_seq_len, n_kv_heads, head_dim]
      cache_seqlens:    [B] (int32), current cache length per batch element.
    """

    def __init__(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.cache_seqlens = cache_seqlens

    @property
    def num_layers(self) -> int:
        return int(self.k_cache.size(0))

    @property
    def max_seq_len(self) -> int:
        return int(self.k_cache.size(2))

    @property
    def batch_size(self) -> int:
        return int(self.k_cache.size(1))

    @staticmethod
    def allocate(
        *,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        k = torch.empty(
            (num_layers, batch_size, max_seq_len, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        v = torch.empty_like(k)
        # Keep KV padding finite once at allocation time.
        # Initializing once at allocation time keeps padding finite without per-step overhead.
        k.zero_()
        v.zero_()
        cache_seqlens = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        return KVCache(k, v, cache_seqlens)

    def reset(self):
        self.cache_seqlens.zero_()

    def advance(self, n: int):
        # cache_seqlens is int32 to keep counters compact and avoid dtype churn.
        self.cache_seqlens += int(n)

    def layer(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[i], self.v_cache[i]


class YarnState:
    """Incremental YaRN + 多軸 MRoPE state for O(1) per-token rotary in decode.

    Yarn.forward() は全文長 T に対して O(T) で軸インデックスを構成するが、
    KV-cache decode では 1 token ごとに O(1) で cos/sin を得たい。

    本 State は forward のロジック（pos/pitch forward-fill, skyline 検出, interval 生成）
    を逐次更新の状態機械として実装し、毎ステップの Python オーバーヘッドを最小化する。

    軸構成（固定 6 分割）:
      (index, pos, pitch_oct, pitch_pc, interval_same, interval_prev)
    """

    def __init__(self, yarn: "Yarn", *, device: torch.device, dtype: torch.dtype):
        self.yarn = yarn
        self.device = device
        self.dtype = dtype

        y = yarn
        # Snapshot caches in the inference dtype to avoid fp16/bf16 mixing (-> fp32 upcast).
        self._cos_index = y._cos_index.to(device=device, dtype=dtype)
        self._sin_index = y._sin_index.to(device=device, dtype=dtype)
        self._cos_pos = y._cos_pos.to(device=device, dtype=dtype)
        self._sin_pos = y._sin_pos.to(device=device, dtype=dtype)
        self._cos_pitch = y._cos_pitch.to(device=device, dtype=dtype)
        self._sin_pitch = y._sin_pitch.to(device=device, dtype=dtype)
        self._cos_ivl_same = y._cos_ivl_same.to(device=device, dtype=dtype)
        self._sin_ivl_same = y._sin_ivl_same.to(device=device, dtype=dtype)
        self._cos_ivl_prev = y._cos_ivl_prev.to(device=device, dtype=dtype)
        self._sin_ivl_prev = y._sin_ivl_prev.to(device=device, dtype=dtype)

        # Preallocate per-step output buffers to avoid clone() allocations in decode.
        total_pairs = int(self._cos_index.size(1))
        self._cos_step = torch.empty((1, 1, 1, total_pairs), device=device, dtype=dtype)
        self._sin_step = torch.empty((1, 1, 1, total_pairs), device=device, dtype=dtype)
        self._cos_step_1d = self._cos_step.view(-1)
        self._sin_step_1d = self._sin_step.view(-1)

        self.reset()

    def reset(self):
        self.idx_pos = 0  # absolute token index (index axis)
        self.pos_state = 0
        self.pitch_state = 0

        # skyline / interval state (all offset-coded: 0 means "none")
        self.prev_skyline_id = 0       # last observed skyline pitch id (pitch_abs+1)
        self.seg_prev_skyline_id = 0   # snapshot at POS: previous skyline for this segment
        self.skyline_cur_id = 0        # skyline pitch id for current POS group (0 until first pitch)
        self.expect_skyline = False    # True after POS until first pitch token

    @torch.no_grad()
    def init_from_prompt(self, token_ids: torch.Tensor):
        """Initialize state after consuming an existing prefix (prompt).

        token_ids: [T] int64/long tensor on device.
        """
        if token_ids.numel() == 0:
            self.reset()
            return

        tok = token_ids.to(torch.int32)
        T = int(tok.numel())
        idx = torch.arange(T, device=tok.device, dtype=torch.int32)

        y = self.yarn
        pitch_tok = (tok >= int(y.pitch_start)) & (tok < int(y.pitch_start + y.pitch_size))
        pos_tok = (tok >= int(y.pos_start)) & (tok < int(y.pos_start + y.pos_size))
        bar_tok = tok.eq(int(y.bar_id)) if y.bar_id is not None else torch.zeros_like(tok, dtype=torch.bool)
        doc_tok = tok.eq(int(y.doc_id)) if y.doc_id is not None else torch.zeros_like(tok, dtype=torch.bool)

        # pos axis state (already includes offset)
        pos_updates = torch.where(
            pos_tok,
            (tok - int(y.pos_start) + int(y.pos_idx_offset)).to(torch.int32),
            torch.zeros_like(tok, dtype=torch.int32),
        )
        pos_update_mask = pos_tok | bar_tok | doc_tok
        pos_idx = forward_fill_from_updates(pos_updates, pos_update_mask, idx)
        self.pos_state = int(pos_idx[-1].item())

        # pitch axis state (pre-gate)
        pitch_updates = torch.where(
            pitch_tok,
            (tok - int(y.pitch_start) + int(y.pitch_idx_offset)).to(torch.int32),
            torch.zeros_like(tok, dtype=torch.int32),
        )
        if bool(y.carry_pitch):
            pitch_update_mask = pitch_tok | doc_tok
            if bool(y.reset_pitch_on_pos):
                pitch_update_mask = pitch_update_mask | pos_tok
            if bool(y.reset_pitch_on_bar):
                pitch_update_mask = pitch_update_mask | bar_tok
            pitch_idx = forward_fill_from_updates(pitch_updates, pitch_update_mask, idx)
        else:
            pitch_idx = pitch_updates
        self.pitch_state = int(pitch_idx[-1].item())

        # skyline token: first pitch immediately following a POS token
        prev_is_pos = torch.cat([pos_tok.new_zeros((1,)), pos_tok[:-1]], dim=0)
        skyline_tok = pitch_tok & prev_is_pos
        pitch_abs = (tok - int(y.pitch_start)).to(torch.int32)

        # last skyline pitch id across the stream (offset-coded), reset on doc
        skyline_id_updates = torch.where(skyline_tok, pitch_abs + 1, torch.zeros_like(pitch_abs))
        last_skyline_id = forward_fill_from_updates(skyline_id_updates, skyline_tok | doc_tok, idx)
        self.prev_skyline_id = int(last_skyline_id[-1].item())

        # per-segment previous skyline id snapshot (updated at POS)
        prev_seg_updates = torch.where(pos_tok, last_skyline_id, torch.zeros_like(pitch_abs))
        seg_prev_id = forward_fill_from_updates(prev_seg_updates, pos_tok | bar_tok | doc_tok, idx)
        self.seg_prev_skyline_id = int(seg_prev_id[-1].item())

        # current segment skyline id (updated at skyline_tok, reset at pos/bar/doc)
        cur_updates = torch.where(skyline_tok, pitch_abs + 1, torch.zeros_like(pitch_abs))
        cur_id = forward_fill_from_updates(cur_updates, skyline_tok | pos_tok | bar_tok | doc_tok, idx)
        self.skyline_cur_id = int(cur_id[-1].item())

        # expect_skyline: last triggering event among {doc/bar/pos/skyline}
        idx_i64 = idx.to(torch.int64)
        last_pos = torch.where(pos_tok, idx_i64, torch.full_like(idx_i64, -1)).max().item()
        last_sky = torch.where(skyline_tok, idx_i64, torch.full_like(idx_i64, -1)).max().item()
        last_reset = torch.where((bar_tok | doc_tok), idx_i64, torch.full_like(idx_i64, -1)).max().item()
        self.expect_skyline = bool(last_pos > max(last_sky, last_reset))

        # index axis absolute position
        self.idx_pos = T

    @torch.no_grad()
    def step(self, tok_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consume one token id and return (cos, sin) for that token position.

        Returns:
          cos, sin: [1, 1, 1, head_dim//2] in self.dtype on device.
        """
        y = self.yarn
        t = int(self.idx_pos)
        tok = int(tok_id)

        # token type flags
        pitch_tok = (tok >= int(y.pitch_start)) and (tok < int(y.pitch_start + y.pitch_size))
        pos_tok = (tok >= int(y.pos_start)) and (tok < int(y.pos_start + y.pos_size))
        bar_tok = (tok == int(y.bar_id)) if (y.bar_id is not None) else False
        doc_tok = (tok == int(y.doc_id)) if (y.doc_id is not None) else False

        # --- update states (pos/pitch/skyline) ---

        # POS axis state (forward-fill with resets on bar/doc)
        if doc_tok or bar_tok:
            self.pos_state = 0
        elif pos_tok:
            self.pos_state = tok - int(y.pos_start) + int(y.pos_idx_offset)

        # Pitch axis state (pre-gate)
        if not bool(y.carry_pitch):
            self.pitch_state = (tok - int(y.pitch_start) + int(y.pitch_idx_offset)) if pitch_tok else 0
        else:
            reset = doc_tok
            if bool(y.reset_pitch_on_bar) and bar_tok:
                reset = True
            if bool(y.reset_pitch_on_pos) and pos_tok:
                reset = True
            if reset:
                self.pitch_state = 0
            if pitch_tok:
                self.pitch_state = tok - int(y.pitch_start) + int(y.pitch_idx_offset)

        # Skyline / interval state machine
        if doc_tok:
            self.prev_skyline_id = 0
            self.seg_prev_skyline_id = 0
            self.skyline_cur_id = 0
            self.expect_skyline = False
        elif bar_tok:
            # bar は pos/pitch の reset に加え、同一 POS 群の skyline も終了させる
            self.seg_prev_skyline_id = 0
            self.skyline_cur_id = 0
            self.expect_skyline = False
        elif pos_tok:
            # new segment start: snapshot previous skyline for interval_prev
            self.seg_prev_skyline_id = int(self.prev_skyline_id)
            self.skyline_cur_id = 0
            self.expect_skyline = True

        # interval indices (default: non pitch)
        ivl_same_idx = 0
        ivl_prev_idx = 0

        if pitch_tok:
            pitch_abs = tok - int(y.pitch_start)  # 0..pitch_size-1

            # detect skyline (first pitch after POS)
            if self.expect_skyline:
                self.skyline_cur_id = pitch_abs + 1
                self.prev_skyline_id = self.skyline_cur_id
                self.expect_skyline = False

            # interval_same: diff to current skyline (skyline itself => 0 diff, but not non pitch)
            if self.skyline_cur_id > 0:
                skyline_abs = self.skyline_cur_id - 1
                diff_same = pitch_abs - skyline_abs
                ivl_same_idx = diff_same - int(y.ivl_same_min) + int(y.ivl_idx_offset)
                if ivl_same_idx < 0:
                    ivl_same_idx = 0
                if ivl_same_idx >= self._cos_ivl_same.size(0):
                    ivl_same_idx = self._cos_ivl_same.size(0) - 1

            # interval_prev: diff to previous skyline (piece start => non pitch)
            if self.seg_prev_skyline_id > 0:
                prev_abs = self.seg_prev_skyline_id - 1
                diff_prev = pitch_abs - prev_abs
                ivl_prev_idx = diff_prev - int(y.ivl_prev_min) + int(y.ivl_idx_offset)
                if ivl_prev_idx < 0:
                    ivl_prev_idx = 0
                if ivl_prev_idx >= self._cos_ivl_prev.size(0):
                    ivl_prev_idx = self._cos_ivl_prev.size(0) - 1
            else:
                ivl_prev_idx = 0

        # apply pitch gate (output index only)
        if getattr(y, "pitch_gate", "pitch_tok_only") == "pitch_tok_only" and (not pitch_tok):
            pitch_idx = 0
        else:
            pitch_idx = int(self.pitch_state)

        # clamp indices
        pos_idx = int(self.pos_state)
        if pos_idx < 0:
            pos_idx = 0
        if pos_idx >= self._cos_pos.size(0):
            pos_idx = self._cos_pos.size(0) - 1

        if pitch_idx < 0:
            pitch_idx = 0
        if pitch_idx >= self._cos_pitch.size(0):
            pitch_idx = self._cos_pitch.size(0) - 1

        # base index cos/sin (index axis can be disabled -> identity row 0)
        base_t = t if bool(y.enable_index) else 0
        if base_t >= self._cos_index.size(0):
            raise RuntimeError(f"Sequence length {t} exceeds YaRN cache (max_seq_len={self._cos_index.size(0)})")
        cos = self._cos_step_1d
        sin = self._sin_step_1d
        cos.copy_(self._cos_index[base_t])
        sin.copy_(self._sin_index[base_t])

        # overwrite axial sections (same logic as Yarn.forward)
        stride = len(y.mrope_section)
        s0, s1, s2, s3, s4, s5 = y.mrope_section

        # POS axis overwrite (axis_id=1)
        if bool(y.enable_pos) and s1 > 0:
            cos_pos = self._cos_pos[pos_idx]
            sin_pos = self._sin_pos[pos_idx]
            if y.pair_layout == "block":
                o1 = s0
                o2 = s0 + s1
                cos[o1:o2] = cos_pos[o1:o2]
                sin[o1:o2] = sin_pos[o1:o2]
            else:
                end = s1 * stride
                cos[1:end:stride] = cos_pos[1:end:stride]
                sin[1:end:stride] = sin_pos[1:end:stride]

        # pitch axes overwrite (axis_id=2,3) share the same pitch cache row
        cos_pitch = sin_pitch = None
        if bool(y.enable_pitch) and ((s2 > 0) or (s3 > 0)):
            cos_pitch = self._cos_pitch[pitch_idx]
            sin_pitch = self._sin_pitch[pitch_idx]

        # pitch_oct axis overwrite (axis_id=2)
        if cos_pitch is not None and s2 > 0:
            if y.pair_layout == "block":
                o2 = s0 + s1
                o3 = o2 + s2
                cos[o2:o3] = cos_pitch[o2:o3]
                sin[o2:o3] = sin_pitch[o2:o3]
            else:
                end = s2 * stride
                cos[2:end:stride] = cos_pitch[2:end:stride]
                sin[2:end:stride] = sin_pitch[2:end:stride]

        # pitch_pc axis overwrite (axis_id=3)
        if cos_pitch is not None and s3 > 0:
            if y.pair_layout == "block":
                o3 = s0 + s1 + s2
                o4 = o3 + s3
                cos[o3:o4] = cos_pitch[o3:o4]
                sin[o3:o4] = sin_pitch[o3:o4]
            else:
                end = s3 * stride
                cos[3:end:stride] = cos_pitch[3:end:stride]
                sin[3:end:stride] = sin_pitch[3:end:stride]

        # interval_same axis overwrite (axis_id=4)
        if bool(y.enable_ivl_same) and s4 > 0:
            cos_is = self._cos_ivl_same[int(ivl_same_idx)]
            sin_is = self._sin_ivl_same[int(ivl_same_idx)]
            if y.pair_layout == "block":
                o4 = s0 + s1 + s2 + s3
                o5 = o4 + s4
                cos[o4:o5] = cos_is[o4:o5]
                sin[o4:o5] = sin_is[o4:o5]
            else:
                end = s4 * stride
                cos[4:end:stride] = cos_is[4:end:stride]
                sin[4:end:stride] = sin_is[4:end:stride]

        # interval_prev axis overwrite (axis_id=5)
        if bool(y.enable_ivl_prev) and s5 > 0:
            cos_ip = self._cos_ivl_prev[int(ivl_prev_idx)]
            sin_ip = self._sin_ivl_prev[int(ivl_prev_idx)]
            if y.pair_layout == "block":
                o5 = s0 + s1 + s2 + s3 + s4
                cos[o5:] = cos_ip[o5:]
                sin[o5:] = sin_ip[o5:]
            else:
                end = s5 * stride
                cos[5:end:stride] = cos_ip[5:end:stride]
                sin[5:end:stride] = sin_ip[5:end:stride]

        self.idx_pos += 1

        return self._cos_step, self._sin_step


# ---- Inference forward path ----

@torch.no_grad()
def _attn_forward_infer(
    attn: CausalSelfAttention,
    x: torch.Tensor,
    ve: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attn_scale: float,
    window_tokens: int,
    block_mask: Optional[BlockMask],
    use_kv_cache: bool,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    cache_start: int,
) -> torch.Tensor:
    B, T, _ = x.size()

    q = attn.c_q(x).view(B, T, attn.num_heads, attn.head_dim)
    k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)

    if ve is not None and attn.ve_gate is not None:
        ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
        gate = 2 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    qh = q.transpose(1, 2)
    gqa = (attn.num_heads != attn.n_kv_head)

    if use_kv_cache:
        assert k_cache is not None and v_cache is not None and cache_seqlens is not None
        assert B == 1, "KV-cache path expects batch size 1"
        start = int(cache_start)
        end = start + T
        k_cache[:, start:end].copy_(k)
        v_cache[:, start:end].copy_(v)

        if T == 1 and start > 0:
            kv_begin = max(0, end - int(window_tokens))
            k_ctx = k_cache[:, kv_begin:end]
            v_ctx = v_cache[:, kv_begin:end]
            y = _flex_attention_call(
                qh,
                k_ctx.transpose(1, 2),
                v_ctx.transpose(1, 2),
                scale=attn_scale,
                enable_gqa=gqa,
            )
        else:
            assert block_mask is not None
            k_ctx = k_cache[:, :end]
            v_ctx = v_cache[:, :end]
            y = _flex_attention_call(
                qh,
                k_ctx.transpose(1, 2),
                v_ctx.transpose(1, 2),
                block_mask=block_mask,
                scale=attn_scale,
                enable_gqa=gqa,
            )
    else:
        assert block_mask is not None
        y = _flex_attention_call(
            qh,
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=attn_scale,
            enable_gqa=gqa,
        )

    y = y.transpose(1, 2)
    y = y.contiguous().view(B, T, attn.num_heads * attn.head_dim)
    y = attn.c_proj(y)
    return y


@torch.no_grad()
def gpt_forward_infer_logits(
    model: GPT,
    input_seq: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    schedule_cfg: ForwardScheduleConfig,
    kv_cache: Optional[KVCache],
    use_kv_cache: bool,
) -> torch.Tensor:
    """Forward pass for inference returning logits.

    input_seq: [T] token ids (int64) on device.
    cos/sin:   [1, T, 1, head_dim//2] in the same dtype as inference autocast.
    Returns:
      logits: [1, T, vocab_size] float32
    """
    assert input_seq.ndim == 1
    ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

    B = 1
    T = int(input_seq.size(0))
    assert T > 0

    x = model.transformer.wte(input_seq)
    x = norm(x)[None]
    x0 = x

    bm_sizes = model._get_bm_sizes(ws_short, ws_long)
    cache_start = 0
    if kv_cache is not None and use_kv_cache:
        cache_start = int(kv_cache.cache_seqlens[0].item())

    block_masks_by_window: Optional[dict[int, BlockMask]] = None
    need_block_masks = not (kv_cache is not None and use_kv_cache and cache_start > 0 and T == 1)
    if need_block_masks:
        window_set = set(int(x) for x in bm_sizes)
        if kv_cache is None or (not use_kv_cache) or cache_start == 0:
            infer_seqlens = torch.tensor([0, T], device=input_seq.device, dtype=torch.int32)
            block_masks_by_window = _build_varlen_flex_block_masks(
                seqlens=infer_seqlens,
                total_len=T,
                window_sizes_tokens=window_set,
                block_size=int(args.block_size),
                device=input_seq.device,
            )
        else:
            kv_len = cache_start + T
            block_masks_by_window = {
                w: _build_causal_window_block_mask(
                    q_len=T,
                    kv_len=kv_len,
                    q_start=cache_start,
                    window_tokens=w,
                    block_size=int(args.block_size),
                    device=input_seq.device,
                )
                for w in window_set
            }

    # Cast per-layer scalars once and cache for KV-cache decode (avoids per-token allocations).
    a_vec = getattr(model, "_infer_resid_lambdas_cache", None)
    b_vec = getattr(model, "_infer_x0_lambdas_cache", None)
    if (
        a_vec is None
        or b_vec is None
        or a_vec.device != x.device
        or b_vec.device != x.device
        or a_vec.dtype != x.dtype
        or b_vec.dtype != x.dtype
        or a_vec.numel() != model.resid_lambdas.numel()
        or b_vec.numel() != model.x0_lambdas.numel()
    ):
        a_vec = model.resid_lambdas.to(device=x.device, dtype=x.dtype)
        b_vec = model.x0_lambdas.to(device=x.device, dtype=x.dtype)
        model._infer_resid_lambdas_cache = a_vec
        model._infer_x0_lambdas_cache = b_vec

    for i, block in enumerate(model.transformer.h):
        a = a_vec[i]
        b = b_vec[i]
        x = a * x + b * x0
        ve = model.value_embeds[str(i)](input_seq) if str(i) in model.value_embeds else None

        if kv_cache is not None and use_kv_cache:
            k_cache_i, v_cache_i = kv_cache.layer(i)
            cache_seqlens = kv_cache.cache_seqlens
        else:
            k_cache_i = v_cache_i = cache_seqlens = None

        block_mask_i = None if block_masks_by_window is None else block_masks_by_window[bm_sizes[i]]
        # attention
        x = x + _attn_forward_infer(
            block.attn,
            norm(x),
            ve,
            cos[:, :T],
            sin[:, :T],
            attn_scale=model.yarn.attn_scale,
            window_tokens=int(bm_sizes[i]),
            block_mask=block_mask_i,
            use_kv_cache=(kv_cache is not None and use_kv_cache),
            k_cache=k_cache_i,
            v_cache=v_cache_i,
            cache_seqlens=cache_seqlens,
            cache_start=cache_start,
        )
        # MLP
        x = x + block.mlp(norm(x))

    x = norm(x)
    logits = model.lm_head(x)
    logits = logits[..., : model.vocab_size]
    softcap = 15.0
    logits = softcap * torch.tanh(logits.float() / softcap)
    return logits


# ---- MIDI <-> token helpers (REMI96 encoder + model-vocab shift) ----

def _parse_slice(spec: str, length: int) -> slice:
    # spec like "start:end" where either can be empty; supports negative.
    if spec is None or spec == "" or spec == ":":
        return slice(None, None, None)
    if ":" not in spec:
        # single index -> [i:i+1]
        i = int(spec)
        return slice(i, i + 1, None)
    s0, s1 = spec.split(":", 1)
    start = int(s0) if s0.strip() != "" else None
    end = int(s1) if s1.strip() != "" else None
    sl = slice(start, end, None)
    # normalize negatives for safety
    a, b, _ = sl.indices(length)
    return slice(a, b, None)


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Strip common wrapper prefixes from checkpoint state_dict keys.

    Handles:
      - DDP:           'module.'
      - torch.compile: '_orig_mod.'
    Applies repeatedly until no prefix matches remain.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ("module.", "_orig_mod.")
    changed = True
    sd = state_dict
    while changed:
        changed = False
        for p in prefixes:
            if any(k.startswith(p) for k in sd.keys()):
                sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}
                changed = True
    return sd


def _ensure_bos_bar_pos0(
    tokens: Sequence[int],
    *,
    bos_id: int,
    bar_id: int,
    pos0_id: int,
    pos_offset: int,
    pos_size: int,
) -> list[int]:
    out = list(tokens)
    if len(out) == 0 or out[0] != bos_id:
        out = [bos_id] + out
    # ensure next is BAR
    if len(out) == 1 or out[1] != bar_id:
        out.insert(1, bar_id)
    # ensure next is a POS token (avoid duplicate POS0 when slice already starts with POS)
    if len(out) == 2 or not (pos_offset <= int(out[2]) < pos_offset + pos_size):
        out.insert(2, pos0_id)
    return out


def _strip_trailing_special(tokens: Sequence[int], *, eof_id: int, pad_id: int) -> list[int]:
    out = list(tokens)
    while out and (out[-1] == eof_id or out[-1] == pad_id):
        out.pop()
    return out


_REMI_STATE_AFTER_BOS = 0
_REMI_STATE_AFTER_BAR = 1
_REMI_STATE_AFTER_POS = 2
_REMI_STATE_AFTER_PITCH = 3


def _remi_step_state(
    tok: int,
    state: int,
    last_pos: int,
    last_pitch: int | None,
    *,
    remi: Remi96Spec,
    strict: bool,
) -> tuple[int, int, int | None]:
    if tok == remi.pad_id:
        return state, last_pos, last_pitch
    if tok == remi.bos_id:
        if strict:
            raise ValueError("unexpected BOS token in prompt")
        return state, last_pos, last_pitch
    if tok == remi.bar_id:
        if strict and state not in (_REMI_STATE_AFTER_BOS, _REMI_STATE_AFTER_PITCH):
            raise ValueError("BAR must follow BOS or PITCH")
        return _REMI_STATE_AFTER_BAR, -1, None
    if remi.pos_offset <= tok < remi.pos_offset + remi.pos_size:
        pos = int(tok - remi.pos_offset)
        if strict:
            if state not in (_REMI_STATE_AFTER_BAR, _REMI_STATE_AFTER_PITCH):
                raise ValueError("POS must follow BAR or PITCH")
            if state == _REMI_STATE_AFTER_PITCH and pos <= int(last_pos):
                raise ValueError("POS must strictly increase within bar")
        return _REMI_STATE_AFTER_POS, pos, None
    if remi.pitch_min <= tok <= remi.pitch_max:
        if strict and state not in (_REMI_STATE_AFTER_POS, _REMI_STATE_AFTER_PITCH):
            raise ValueError("PITCH requires prior POS")
        if strict and state == _REMI_STATE_AFTER_PITCH and last_pitch is not None and tok >= int(last_pitch):
            raise ValueError("PITCH must be strictly descending within POS")
        return _REMI_STATE_AFTER_PITCH, last_pos, int(tok)
    if strict:
        raise ValueError(f"unknown token id: {tok}")
    return state, last_pos, last_pitch


def _remi_init_state_from_prompt(
    tokens: Sequence[int],
    *,
    remi: Remi96Spec,
    eof_id: int,
    strict: bool,
) -> tuple[int, int, int | None]:
    idx = 0
    n = len(tokens)
    while idx < n and tokens[idx] == remi.pad_id:
        idx += 1
    if idx >= n:
        raise ValueError("prompt is empty after stripping PAD")
    if int(tokens[idx]) != int(remi.bos_id):
        raise ValueError("prompt must start with BOS")
    state = _REMI_STATE_AFTER_BOS
    last_pos = -1
    last_pitch = None
    idx += 1
    while idx < n:
        tok = int(tokens[idx])
        idx += 1
        if tok == remi.pad_id:
            continue
        if tok == int(eof_id):
            # Treat EOF as terminal in the prompt; do not advance state beyond it.
            break
        state, last_pos, last_pitch = _remi_step_state(
            tok, state, last_pos, last_pitch, remi=remi, strict=strict
        )
    return state, last_pos, last_pitch


def _build_remi_constraint_masks(
    *,
    vocab_size: int,
    device: torch.device,
    remi: Remi96Spec,
    eof_id: int,
) -> dict:
    pitch_size = int(remi.pitch_size)
    pos_size = int(remi.pos_size)
    pos_offset = int(remi.pos_offset)
    pitch_min = int(remi.pitch_min)
    pitch_max = int(remi.pitch_max)
    bar_id = int(remi.bar_id)

    mask_pitch_all = torch.zeros((vocab_size,), device=device, dtype=torch.bool)
    mask_pitch_all[pitch_min : pitch_max + 1] = True
    mask_pos_all = torch.zeros((vocab_size,), device=device, dtype=torch.bool)
    mask_pos_all[pos_offset : pos_offset + pos_size] = True
    mask_bar = torch.zeros((vocab_size,), device=device, dtype=torch.bool)
    mask_bar[bar_id] = True
    mask_eof = torch.zeros((vocab_size,), device=device, dtype=torch.bool)
    mask_eof[int(eof_id)] = True

    pos_ids = torch.arange(pos_size, device=device)
    last_pos_vals = torch.arange(-1, pos_size, device=device)
    pos_gt = pos_ids[None, :] > last_pos_vals[:, None]
    mask_pos_gt = torch.zeros((pos_size + 1, vocab_size), device=device, dtype=torch.bool)
    mask_pos_gt[:, pos_offset : pos_offset + pos_size] = pos_gt

    pitch_ids = torch.arange(pitch_size, device=device)
    last_pitch_vals = torch.arange(pitch_size + 1, device=device)
    pitch_lt = pitch_ids[None, :] < last_pitch_vals[:, None]
    mask_pitch_lt = torch.zeros((pitch_size + 1, vocab_size), device=device, dtype=torch.bool)
    mask_pitch_lt[:, pitch_min : pitch_max + 1] = pitch_lt

    return dict(
        pitch_all=mask_pitch_all,
        pos_all=mask_pos_all,
        bar=mask_bar,
        eof=mask_eof,
        pos_gt=mask_pos_gt,
        pitch_lt=mask_pitch_lt,
        pos_size=pos_size,
        pitch_size=pitch_size,
    )


def _remi_state_mask(
    state: int,
    last_pos: int,
    last_pitch: int | None,
    masks: dict,
) -> torch.Tensor:
    pos_size = int(masks["pos_size"])
    pitch_size = int(masks["pitch_size"])
    if state == _REMI_STATE_AFTER_BOS:
        return masks["bar"]
    if state == _REMI_STATE_AFTER_BAR:
        return masks["pos_all"]
    if state == _REMI_STATE_AFTER_POS:
        return masks["pitch_all"]

    # AFTER_PITCH: allow PITCH (strictly descending), POS (strictly increasing), BAR, EOF.
    idx = int(last_pitch) if last_pitch is not None and int(last_pitch) >= 0 else pitch_size
    pitch_mask = masks["pitch_lt"][idx]
    pos_row = int(last_pos) + 1 if last_pos is not None else 0
    if last_pos is not None and int(last_pos) >= pos_size - 1:
        # At POS limit: only BAR or EOF is allowed as the next structural token.
        return pitch_mask | masks["bar"] | masks["eof"]
    pos_mask = masks["pos_gt"][pos_row]
    return pitch_mask | pos_mask | masks["bar"] | masks["eof"]


@torch.no_grad()
def _sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    allowed_mask: Optional[torch.Tensor],
    return_tensor: bool = False,
) -> int | torch.Tensor:
    # logits_1d: [vocab]
    if allowed_mask is not None:
        logits_1d = logits_1d.masked_fill(~allowed_mask, float("-inf"))
    if temperature is None or temperature <= 0.0:
        next_id = torch.argmax(logits_1d)
        return next_id if return_tensor else int(next_id.item())

    x = logits_1d / float(temperature)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), x.size(-1))
        v, _ = torch.topk(x, k)
        cutoff = v[-1]
        x = torch.where(x < cutoff, float("-inf"), x)

    probs = torch.softmax(x, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).squeeze(0)
    return next_id if return_tensor else int(next_id.item())


def _remi96_tokenize_one(
    midi_path: str,
    *,
    tpq: int,
    min_dur: int,
    include_drums: bool,
):
    """Tokenize one MIDI using the user's REMI96 tokenizer script.

    Returns:
      tokens_remi96: np.ndarray[int] (REMI96 id space)
      meta: dict with bar_ticks_reltime, time_shift, tpq, min_dur, etc.
    """
    import importlib.util
    import numpy as np

    tok_py = Path(__file__).resolve().parent / "merge_quantize_ontime_tensor_remi96.py"
    if not tok_py.exists():
        raise FileNotFoundError(f"Tokenizer script not found: {tok_py}")

    spec = importlib.util.spec_from_file_location("remi96_tokenizer", str(tok_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    score = mod.load_score(Path(midi_path))
    score_q = score.resample(tpq=int(tpq), min_dur=int(min_dur))
    pitches, times = mod.collect_pitch_time_arrays(score_q.tracks, bool(include_drums))
    pitches, times, transpose, dropped = mod.transpose_pitches_96(times, pitches)
    times, pitches = mod.sort_and_dedup(times, pitches)
    tokens, time_shift, bar_ticks, ts_num, ts_denom = mod.encode_tokens(times, pitches, score_q)

    meta = dict(
        tpq=int(tpq),
        min_dur=int(min_dur),
        include_drums=bool(include_drums),
        time_shift=int(time_shift),
        bar_ticks_reltime=int(bar_ticks),
        ts_num=int(ts_num),
        ts_denom=int(ts_denom),
        transpose=int(transpose),
        dropped=int(dropped),
        note_count=int(pitches.size),
    )
    return tokens.astype(np.int64), meta


def _model_to_remi96_ids(tokens_model: Sequence[int], *, shift: int) -> list[int]:
    """Convert model-vocab token ids to REMI96 token ids.

    本リポジトリの語彙設計（vocab_size=132）では shift=0 が正しく、この関数は恒等写像になる。
    """
    out = []
    shift = int(shift)
    thr = int(REMI96.bar_id) + shift  # BAR id in model vocab when shift>0
    for t in tokens_model:
        ti = int(t)
        if ti >= thr:
            ti = ti - shift
        out.append(ti)
    return out


def _remi96_ids_to_model(tokens_remi: Sequence[int], *, shift: int) -> list[int]:
    out = []
    for t in tokens_remi:
        ti = int(t)
        if ti >= int(REMI96.bar_id):  # BAR and above in REMI96 mapping
            ti = ti + int(shift)
        out.append(ti)
    return out


def _remi96_tokens_to_midi(
    tokens_remi,
    *,
    out_path: str,
    meta: dict,
    velocity: int = 100,
    duration: int = 1,
):
    import numpy as np
    import symusic.core as core  # 追加（重要）

    PITCH_MIN = REMI96.pitch_min
    PITCH_MAX = REMI96.pitch_max
    BAR_ID = REMI96.bar_id
    POS_OFFSET = REMI96.pos_offset

    bar_ticks = int(meta.get("bar_ticks_reltime", 32))
    tpq = int(meta.get("tpq", 8))
    time_shift = int(meta.get("time_shift", 0))

    bos_id = POS_OFFSET + int(bar_ticks)
    eof_id = int(bos_id) + 1
    pad_id = int(bos_id) + 2

    tokens = [int(t) for t in tokens_remi]
    idx = 0
    while idx < len(tokens) and tokens[idx] == pad_id:
        idx += 1
    if idx >= len(tokens):
        raise ValueError("missing BOS token")
    if tokens[idx] != bos_id:
        raise ValueError("BOS token must appear first")
    idx += 1

    times_list: list[int] = []
    pitches_list: list[int] = []
    current_bar = -1
    current_pos = None
    eof_seen = False

    while idx < len(tokens):
        tok = int(tokens[idx]); idx += 1
        if tok == pad_id:
            continue
        if tok == eof_id:
            eof_seen = True
            break
        if tok == bos_id:
            raise ValueError("unexpected BOS token in stream")
        if tok == BAR_ID:
            current_bar += 1
            current_pos = None
            continue
        if tok >= POS_OFFSET and tok < bos_id:
            current_pos = int(tok) - POS_OFFSET
            if current_pos < 0 or current_pos >= bar_ticks:
                raise ValueError("POS out of range")
            continue
        if tok < PITCH_MIN or tok > PITCH_MAX:
            raise ValueError("PITCH out of range")
        if current_bar < 0 or current_pos is None:
            continue
        abs_time = current_bar * bar_ticks + current_pos
        times_list.append(abs_time)
        pitches_list.append(tok)

    # 空でもMIDIを吐けるように、ScoreTickで作る
    score = core.ScoreTick(int(tpq))
    tr = core.TrackTick()
    tr.name = "track0"
    score.tracks.append(tr)

    if len(times_list) == 0:
        score.dump_midi(out_path)
        return

    times = np.asarray(times_list, dtype=np.int32)
    pitches = np.asarray(pitches_list, dtype=np.int32)

    if int(time_shift) != 0:
        # Restore original absolute time base from tokenized (shifted) times.
        times = times + int(time_shift)

    ticks = times.astype(np.int64)

    # NoteTick(time, duration, pitch, velocity) を使う
    for t, p in zip(ticks.tolist(), pitches.tolist()):
        tr.notes.append(core.NoteTick(int(t), int(duration), int(p), int(velocity)))

    score.dump_midi(out_path)

def _infer_find_latest_step(ckpt_dir: str) -> int:
    """Find latest step in a ckpt_resume directory (looks for latest.txt or model_*.pt)."""
    latest_path = os.path.join(ckpt_dir, "latest.txt")
    if os.path.isfile(latest_path):
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            pass
    steps = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
        base = os.path.basename(p)
        try:
            steps.append(int(base.split("_")[1].split(".")[0]))
        except Exception:
            continue
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(steps)


def _resolve_ckpt_paths(ckpt_arg: str, *, ckpt_step: int) -> tuple[str, Optional[str], Optional[int]]:
    """Resolve ckpt input.

    - If ckpt_arg is a directory: load model_{step:06d}.pt (latest if step<0)
      and also try to read optim_{step:06d}_rank0.pt for yarn_attn_scale + ws state.
    - If ckpt_arg is a file:
        * if it's model_{step}.pt: try to also load sibling optim_{step}_rank0.pt
        * if it's optim_{step}_rankX.pt: try to load sibling model_{step}.pt
        * otherwise: use it directly; no shard metadata.
    Returns: (model_path, shard_path_or_none, resolved_step_or_none)
    """
    if os.path.isdir(ckpt_arg):
        step = int(ckpt_step)
        if step < 0:
            step = _infer_find_latest_step(ckpt_arg)
        model_path = os.path.join(ckpt_arg, f"model_{step:06d}.pt")
        shard_path = os.path.join(ckpt_arg, f"optim_{step:06d}_rank0.pt")
        return model_path, (shard_path if os.path.isfile(shard_path) else None), step

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_arg)) or "."
    base = os.path.basename(ckpt_arg)

    # model_{step}.pt -> also try shard
    if base.startswith("model_") and base.endswith(".pt"):
        step_str = base[len("model_") : -len(".pt")]
        if step_str.isdigit():
            step = int(step_str)
            shard_path = os.path.join(ckpt_dir, f"optim_{step:06d}_rank0.pt")
            return ckpt_arg, (shard_path if os.path.isfile(shard_path) else None), step

    # optim_{step}_rankX.pt -> try sibling model_{step}.pt
    if base.startswith("optim_") and base.endswith(".pt") and "_rank" in base:
        parts = base.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            step = int(parts[1])
            model_path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
            if os.path.isfile(model_path):
                return model_path, ckpt_arg, step

    return ckpt_arg, None, None


def _run_inference_cli():

    import argparse
    import json
    import numpy as np
    import re

    parser = argparse.ArgumentParser(description="KV-cache inference (FlexAttention) + MIDI I/O")
    parser.add_argument("--infer", action="store_true", help="Run inference and exit (do not train).")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path. File (.pt) or ckpt_resume directory (contains model_*.pt).",
    )
    parser.add_argument("--ckpt_step", type=int, default=-1, help="If --ckpt is a directory: which step to load (-1 latest).")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Autocast/KV-cache dtype.")
    parser.add_argument("--midi_in", type=str, required=True, help="Input MIDI path.")
    parser.add_argument("--midi_out", type=str, required=True, help="Output MIDI path.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_range", type=str, default=":", help="Python slice spec 'start:end' over prompt token indices.")
    parser.add_argument("--use_kv_cache", type=int, default=1, help="1=enable KV-cache decode, 0=disable (full recompute).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tpq", type=int, default=REMI96.tpq, help="Fixed to 8 for this model.")
    parser.add_argument("--min_dur", type=int, default=1)
    parser.add_argument("--include_drums", type=int, default=0)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--duration", type=int, default=None, help="Default: use --min_dur.")
    parser.add_argument("--remi_shift", type=int, default=0, help="REMI96->model vocab shift for BAR/POS/BOS/EOF/PAD. vocab_size=132 の場合は 0 を指定してください。")
    parser.add_argument("--save_tokens", type=str, default="", help="Optional: save generated model-token ids to .npy")
    parser.add_argument("--save_meta", type=str, default="", help="Optional: save meta json (includes tokenizer meta + generation args).")
    parser.add_argument("--ws_short", type=int, default=-1, help="Short window size (in blocks). Default: from ckpt (if available) else args.ws_final")
    parser.add_argument("--ws_long", type=int, default=-1, help="Long window size (in blocks). Default: from ckpt (if available) else args.ws_final")
    parser.add_argument("--eof_id", type=int, default=REMI96.eof_id, help="EOF token id in model vocab (default 130).")
    parser.add_argument("--disallow_bos_pad", type=int, default=1, help="Disallow BOS/PAD during sampling (recommended).")
    parser.add_argument("--enforce_remi_rules", type=int, default=1, help="Enforce BAR/POS/PITCH ordering and monotonic rules.")

    a = parser.parse_args()
    if not a.infer:
        raise RuntimeError("--infer flag missing (internal).")

    if int(a.tpq) != int(REMI96.tpq):
        raise RuntimeError("This model assumes 4/4 with tpq=8 (bar_ticks=32). Please use --tpq 8.")

    if a.duration is None:
        a.duration = int(a.min_dur)

    # Vocabulary sanity: this script assumes vocab_size=132 (REMI96 + POS32 + BOS/EOF/PAD).
    if int(a.remi_shift) != 0:
        raise RuntimeError("vocab_size=132 の語彙では --remi_shift は 0 を指定してください。")
    if int(a.eof_id) != int(REMI96.eof_id):
        # Allow override, but keep an explicit guardrail because it also defines the stop condition.
        raise RuntimeError("vocab_size=132 の語彙では EOF は 130 です。--eof_id 130 を指定してください。")

    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    device = torch.device(a.device)
    if device.type != "cuda":
        raise RuntimeError("FlexAttention inference requires a CUDA device.")
    torch.cuda.set_device(device)
    globals()["device"] = device  # Yarn.reset() uses global 'device' in this codebase

    # Dtype selection (FlexAttention kernels use bf16/fp16 here)
    if a.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif a.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    # Tokenize MIDI -> REMI96 ids
    tokens_remi, meta = _remi96_tokenize_one(
        a.midi_in,
        tpq=int(a.tpq),
        min_dur=int(a.min_dur),
        include_drums=bool(a.include_drums),
    )
    bar_ticks = int(meta.get("bar_ticks_reltime", 0))
    if bar_ticks != int(REMI96.bar_ticks):
        raise RuntimeError(
            f"bar_ticks_reltime={bar_ticks} (expected 32). "
            "このモデルは pos_size=32 のREMI表現を前提にしています。--tpq 8 (4/4) で再トークナイズしてください。"
        )
    ts_num = int(meta.get("ts_num", 0))
    ts_denom = int(meta.get("ts_denom", 0))
    if ts_num != int(REMI96.ts_num) or ts_denom != int(REMI96.ts_denom):
        raise RuntimeError(f"time signature must be 4/4 (got {ts_num}/{ts_denom}).")


    tokens_model = _remi96_ids_to_model(tokens_remi.tolist(), shift=int(a.remi_shift))

    # Range selection
    sl = _parse_slice(a.prompt_range, len(tokens_model))
    tokens_model = tokens_model[sl]

    # Avoid trailing EOF/PAD in the prompt so we can continue generation safely.
    tokens_model = _strip_trailing_special(tokens_model, eof_id=int(a.eof_id), pad_id=int(REMI96.pad_id))

    # Ensure minimal valid prefix for decoding & YaRN resets
    BOS_ID = REMI96.bos_id
    BAR_ID_MODEL = REMI96.bar_id
    POS0_ID_MODEL = REMI96.pos_offset  # pos=0 with offset=97
    tokens_model = _ensure_bos_bar_pos0(
        tokens_model,
        bos_id=BOS_ID,
        bar_id=BAR_ID_MODEL,
        pos0_id=POS0_ID_MODEL,
        pos_offset=int(REMI96.pos_offset),
        pos_size=int(REMI96.pos_size),
    )

    # Resolve checkpoint paths
    model_path, shard_path, resolved_step = _resolve_ckpt_paths(str(a.ckpt), ckpt_step=int(a.ckpt_step))

    # Load weights
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = None
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unrecognized checkpoint format (expected dict state_dict or dict with 'model').")

    state_dict = _strip_state_dict_prefixes(state_dict)

    # Model hyperparams (must match training)
    vocab_size = int(MODEL_CFG.vocab_size)
    head_dim = int(MODEL_CFG.head_dim)  # must match training; used for RoPE + attention reshape

    # Derive num_layers from checkpoint if possible (robust to config tweaks)
    num_layers = int(MODEL_CFG.num_layers)
    try:
        layer_ids = []
        for k in state_dict.keys():
            m = re.match(r"transformer\.h\.(\d+)\.", k)
            if m:
                layer_ids.append(int(m.group(1)))
        if layer_ids:
            num_layers = max(layer_ids) + 1
    except Exception:
        pass

    # Derive model_dim / num_heads / num_kv_heads from checkpoint shapes if possible
    model_dim = int(MODEL_CFG.model_dim)
    num_heads = int(MODEL_CFG.num_heads)
    num_kv_heads = None

    try:
        wte_w = state_dict.get("transformer.wte.weight", None)
        if wte_w is not None:
            model_dim = int(wte_w.shape[1])
    except Exception:
        pass

    try:
        q_w = state_dict.get("transformer.h.0.attn.c_q.weight", None)
        if q_w is not None:
            q_out = int(q_w.shape[0])
            if q_out % head_dim != 0:
                raise RuntimeError(f"c_q out_features={q_out} not divisible by head_dim={head_dim}")
            num_heads = q_out // head_dim
    except Exception:
        pass

    try:
        k_w = state_dict.get("transformer.h.0.attn.c_k.weight", None)
        if k_w is not None:
            k_out = int(k_w.shape[0])
            if k_out % head_dim != 0:
                raise RuntimeError(f"c_k out_features={k_out} not divisible by head_dim={head_dim}")
            num_kv_heads = k_out // head_dim
    except Exception:
        num_kv_heads = None

    if num_kv_heads is None:
        # Fallback to script defaults
        num_kv_heads = int(args.num_kv_heads) if int(args.num_kv_heads) != 0 else int(num_heads)

    model = GPT(
        vocab_size=vocab_size,
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        model_dim=int(model_dim),
        max_seq_len=args.train_max_seq_len,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # strict=False allows minor key drift; keep this explicit for debugging.
        print("[load_state_dict] missing keys:", missing)
        print("[load_state_dict] unexpected keys:", unexpected)

    # If we have a ckpt_resume shard, restore non-state_dict scalars (yarn_attn_scale) and ws state.
    shard = None
    if shard_path is not None:
        try:
            shard = torch.load(shard_path, map_location="cpu")
        except Exception as e:
            print(f"[infer] warning: failed to load shard file: {shard_path}: {e}")
            shard = None
    if isinstance(shard, dict):
        if "yarn_attn_scale" in shard:
            model.yarn.attn_scale = float(shard["yarn_attn_scale"])
        tm = shard.get("training_manager", {}) if isinstance(shard.get("training_manager", {}), dict) else {}
    else:
        tm = {}

    model.eval()

    # Ensure embeddings are in the chosen low-precision dtype (training default is bf16)
    model.transformer.wte.to(dtype=amp_dtype)
    for ve in model.value_embeds.values():
        ve.to(dtype=amp_dtype)

    # Schedule config
    if int(a.ws_short) > 0:
        ws_short = int(a.ws_short)
    else:
        ws_short = int(tm.get("ws_short", args.ws_final))
    if int(a.ws_long) > 0:
        ws_long = int(a.ws_long)
    else:
        ws_long = int(tm.get("ws_long", args.ws_final))
    schedule_cfg = ForwardScheduleConfig(ws_short=int(ws_short), ws_long=int(ws_long))

    # YaRN キャッシュを checkpoint の inv_freq と同期（過去 ckpt の pos/pitch 未更新も補正）
    model.yarn.apply(int(ws_long), int(ws_long))

    # Build allowed mask
    # vocab_size=132:
    #   0..95  : PITCH
    #   96     : BAR
    #   97..128: POS (32)
    #   129    : BOS
    #   130    : EOF
    #   131    : PAD
    PAD_ID = REMI96.pad_id

    allowed_mask_base = None

    if int(a.disallow_bos_pad) == 1:
        allowed = torch.ones((vocab_size,), device=device, dtype=torch.bool)
        allowed[BOS_ID] = False
        allowed[PAD_ID] = False
        allowed_mask_base = allowed

    enforce_remi = bool(int(a.enforce_remi_rules))
    remi_masks = None
    remi_state = None
    remi_last_pos = None
    remi_last_pitch = None
    if enforce_remi:
        remi_masks = _build_remi_constraint_masks(
            vocab_size=vocab_size,
            device=device,
            remi=REMI96,
            eof_id=int(a.eof_id),
        )
        remi_state, remi_last_pos, remi_last_pitch = _remi_init_state_from_prompt(
            tokens_model,
            remi=REMI96,
            eof_id=int(a.eof_id),
            strict=True,
        )

    # Prepare prompt tensor
    prompt = torch.tensor(tokens_model, device=device, dtype=torch.long)
    prompt_len = int(prompt.numel())
    if prompt_len >= args.train_max_seq_len:
        raise RuntimeError(f"Prompt length {prompt_len} exceeds model max_seq_len {args.train_max_seq_len}")

    use_kv_cache = bool(int(a.use_kv_cache))

    # Generation bookkeeping
    generated: list[int] = list(tokens_model)
    eof_id = int(a.eof_id)

    if use_kv_cache:
        # Allocate KV cache for full (prompt + generation) sequence (bounded by model max_seq_len)
        max_total = min(int(args.train_max_seq_len), prompt_len + int(a.max_new_tokens) + 1)
        kv_cache = KVCache.allocate(
            num_layers=num_layers,
            batch_size=1,
            max_seq_len=max_total,
            n_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            device=device,
            dtype=amp_dtype,
        )
        kv_cache.reset()

        # Prefill
        with torch.no_grad(), autocast_ctx:
            cos, sin = model.yarn(prompt)
            cos = cos.to(dtype=amp_dtype)
            sin = sin.to(dtype=amp_dtype)
            logits = gpt_forward_infer_logits(
                model,
                prompt,
                cos=cos,
                sin=sin,
                schedule_cfg=schedule_cfg,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
        kv_cache.advance(prompt_len)

        # Initialize incremental YaRN state from the prompt (O(T) on GPU, no per-step scan)
        yarn_state = YarnState(model.yarn, device=device, dtype=amp_dtype)
        yarn_state.init_from_prompt(prompt)

        last_logits = logits[0, -1]

        # Decode loop
        for _ in range(int(a.max_new_tokens)):
            step_allowed = allowed_mask_base
            if enforce_remi:
                state_mask = _remi_state_mask(remi_state, remi_last_pos, remi_last_pitch, remi_masks)
                step_allowed = state_mask if step_allowed is None else (step_allowed & state_mask)
            next_id_t = _sample_next_token(
                last_logits,
                temperature=float(a.temperature),
                top_k=int(a.top_k),
                allowed_mask=step_allowed,
                return_tensor=True,
            )
            next_id = int(next_id_t.item())
            generated.append(next_id)
            if enforce_remi and next_id != eof_id:
                remi_state, remi_last_pos, remi_last_pitch = _remi_step_state(
                    next_id,
                    remi_state,
                    remi_last_pos,
                    remi_last_pitch,
                    remi=REMI96,
                    strict=True,
                )
            if next_id == eof_id:
                break
            if len(generated) >= max_total:
                break

            cos1, sin1 = yarn_state.step(next_id)
            tok1 = next_id_t.view(1)
            with torch.no_grad(), autocast_ctx:
                logits1 = gpt_forward_infer_logits(
                    model,
                    tok1,
                    cos=cos1,
                    sin=sin1,
                    schedule_cfg=schedule_cfg,
                    kv_cache=kv_cache,
                    use_kv_cache=True,
                )
            kv_cache.advance(1)
            last_logits = logits1[0, -1]

    else:
        # Full recompute mode (no KV cache): uses FlexAttention kernels.
        with torch.no_grad(), autocast_ctx:
            cos, sin = model.yarn(prompt)
            cos = cos.to(dtype=amp_dtype)
            sin = sin.to(dtype=amp_dtype)
            logits = gpt_forward_infer_logits(
                model,
                prompt,
                cos=cos,
                sin=sin,
                schedule_cfg=schedule_cfg,
                kv_cache=None,
                use_kv_cache=False,
            )
        last_logits = logits[0, -1]

        for _ in range(int(a.max_new_tokens)):
            step_allowed = allowed_mask_base
            if enforce_remi:
                state_mask = _remi_state_mask(remi_state, remi_last_pos, remi_last_pitch, remi_masks)
                step_allowed = state_mask if step_allowed is None else (step_allowed & state_mask)
            next_id_t = _sample_next_token(
                last_logits,
                temperature=float(a.temperature),
                top_k=int(a.top_k),
                allowed_mask=step_allowed,
                return_tensor=True,
            )
            next_id = int(next_id_t.item())
            generated.append(next_id)
            if enforce_remi and next_id != eof_id:
                remi_state, remi_last_pos, remi_last_pitch = _remi_step_state(
                    next_id,
                    remi_state,
                    remi_last_pos,
                    remi_last_pitch,
                    remi=REMI96,
                    strict=True,
                )
            if next_id == eof_id:
                break
            # recompute on full sequence
            seq = torch.tensor(generated, device=device, dtype=torch.long)
            with torch.no_grad(), autocast_ctx:
                cos, sin = model.yarn(seq)
                cos = cos.to(dtype=amp_dtype)
                sin = sin.to(dtype=amp_dtype)
                logits = gpt_forward_infer_logits(
                    model,
                    seq,
                    cos=cos,
                    sin=sin,
                    schedule_cfg=schedule_cfg,
                    kv_cache=None,
                    use_kv_cache=False,
                )
            last_logits = logits[0, -1]

    # Optionally save tokens / meta
    if a.save_tokens:
        np.save(a.save_tokens, np.asarray(generated, dtype=np.int64))
    if a.save_meta:
        out_meta = dict(meta)
        out_meta.update(
            dict(
                ckpt=str(a.ckpt),
                ckpt_step=int(resolved_step) if resolved_step is not None else None,
                ckpt_model_path=str(model_path),
                ckpt_shard_path=str(shard_path) if shard_path is not None else None,
                max_new_tokens=int(a.max_new_tokens),
                prompt_range=str(a.prompt_range),
                use_kv_cache=bool(use_kv_cache),
                temperature=float(a.temperature),
                top_k=int(a.top_k),
                seed=int(a.seed),
                ws_short=int(ws_short),
                ws_long=int(ws_long),
                remi_shift=int(a.remi_shift),
                yarn_attn_scale=float(model.yarn.attn_scale),
                dtype=str(a.dtype),
                device=str(a.device),
            )
        )
        with open(a.save_meta, "w", encoding="utf-8") as f:
            json.dump(out_meta, f, ensure_ascii=False, indent=2)

    # Decode -> MIDI
    # Ensure EOF exists for decoding convenience
    if eof_id not in generated:
        generated_for_decode = generated + [eof_id]
    else:
        generated_for_decode = generated

    tokens_remi_out = _model_to_remi96_ids(generated_for_decode, shift=int(a.remi_shift))
    _remi96_tokens_to_midi(
        tokens_remi_out,
        out_path=str(a.midi_out),
        meta=meta,
        velocity=int(a.velocity),
        duration=int(a.duration),
    )

    print(f"[infer] prompt_tokens={prompt_len} generated_total={len(generated)} use_kv_cache={use_kv_cache}")
    print(f"[infer] ckpt_model={model_path} ckpt_step={resolved_step}")
    if shard_path is not None:
        print(f"[infer] ckpt_shard={shard_path} yarn_attn_scale={model.yarn.attn_scale}")
    print(f"[infer] wrote: {a.midi_out}")


# If invoked with --infer, run inference and exit before any distributed-training side effects.
if __name__ == "__main__" and "--infer" in sys.argv:
    _run_inference_cli()
    raise SystemExit(0)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
tokens_per_fwdbwd = args.device_batch_size_tokens
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size

def compute_grad_accum_steps(total_tokens: int) -> int:
    assert total_tokens % world_tokens_per_fwdbwd == 0, (
        "Total batch size must be divisible by device batch tokens * world size"
    )
    return total_tokens // world_tokens_per_fwdbwd

grad_accum_steps = compute_grad_accum_steps(args.total_batch_size_tokens)
val_grad_accum_steps = compute_grad_accum_steps(args.val_batch_size)
assert args.device_batch_size_tokens <= args.train_max_seq_len, "device_batch_size_tokens must be <= train_max_seq_len"
assert args.device_batch_size_tokens % 16 == 0, (
    "device_batch_size_tokens must be a multiple of 16"
)
max_ws = max(args.ws_schedule + (args.ws_final, args.ws_validate_post_yarn_ext))
max_window_tokens = max_ws * args.block_size
assert args.device_batch_size_tokens >= max_window_tokens, (
    f"device_batch_size_tokens ({args.device_batch_size_tokens}) must cover max window "
    f"{max_ws} * block_size {args.block_size} = {max_window_tokens}"
)

# Validate batch sizes for all scheduled steps up-front
_all_train_bs = list(args.train_bs_schedule) + [args.train_bs_extension]
for bs in _all_train_bs + [args.val_batch_size]:
    assert bs % world_tokens_per_fwdbwd == 0, (
        f"Batch size {bs} must be divisible by device_batch_size_tokens * world_size "
        f"({world_tokens_per_fwdbwd})"
    )
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Ensure a single run_id across all ranks (Hyperparameters.run_id default is random per process)
_run_id_obj = [args.run_id] if master_process else [None]
dist.broadcast_object_list(_run_id_obj, src=0, device=device)
args.run_id = _run_id_obj[0]
run_id = args.run_id

# Default checkpoint directory (shared across ranks)
if args.checkpoint_dir == "":
    args.checkpoint_dir = os.path.join("logs", run_id)

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)
print0(f"Tokens / micro-batch / rank: {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size_tokens:,} => gradient accumulation steps: {grad_accum_steps}")

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

num_layers = int(MODEL_CFG.num_layers)
num_heads = int(MODEL_CFG.num_heads)
head_dim = int(MODEL_CFG.head_dim)
model_dim = int(MODEL_CFG.model_dim)
model: nn.Module = GPT(
    vocab_size=int(MODEL_CFG.vocab_size),
    num_layers=num_layers,
    num_heads=num_heads,
    num_kv_heads=(args.num_kv_heads or num_heads),
    head_dim=head_dim,
    model_dim=model_dim,
    max_seq_len=args.train_max_seq_len
).cuda()
if model.transformer.wte.weight.device.type == "cuda":
    model.transformer.wte.to(dtype=torch.bfloat16)
    for ve in model.value_embeds.values():
        ve.to(dtype=torch.bfloat16)
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

#model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
model: nn.Module = model
training_manager = TrainingManager(model)

# -----------------------------------------------------------------------------
# Checkpointing (nanochat-style, sharded optimizer states)

def _ckpt_paths(ckpt_dir: str, step: int, rank: int):
    model_path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
    optim_path = os.path.join(ckpt_dir, f"optim_{step:06d}_rank{rank}.pt")
    meta_path = os.path.join(ckpt_dir, f"meta_{step:06d}.json")
    latest_path = os.path.join(ckpt_dir, "latest.txt")
    return model_path, optim_path, meta_path, latest_path


def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_write_text(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _atomic_write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _find_latest_step(ckpt_dir: str) -> int:
    # fast path
    latest_path = os.path.join(ckpt_dir, "latest.txt")
    if os.path.isfile(latest_path):
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            pass
    # fallback: scan model checkpoints
    steps = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
        base = os.path.basename(p)
        try:
            steps.append(int(base.split("_")[1].split(".")[0]))
        except Exception:
            continue
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(steps)


def _to_device(obj, device: torch.device):
    """Recursively move tensors to a given device (used for optimizer state)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj


def save_checkpoint(step: int, *, model: nn.Module, training_manager: TrainingManager, train_loader: ResumableDistributedDataLoader):
    """Save a resumable checkpoint at a *step boundary*.

    We follow nanochat's convention: model weights are saved once (rank 0), while optimizer
    states are saved per rank because they are sharded.
    """
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save/load state dicts from the underlying module when torch.compile is used.
    model_for_state = model._orig_mod if hasattr(model, "_orig_mod") else model

    # per-rank shard: optimizer(s) + training/dataloader/rng state
    shard = {
        "version": 1,
        "step": int(step),
        "world_size": int(world_size),
        "rank": int(rank),
        "optimizers": [opt.state_dict() for opt in training_manager.optimizers],
        "training_manager": {
            "ws_short": int(training_manager.ws_short),
            "ws_long": int(training_manager.ws_long),
            "batch_size": int(training_manager.batch_size),
            "grad_accum_steps": int(training_manager.grad_accum_steps),
            "batch_lr_scale": float(training_manager.batch_lr_scale),
            "weight_decay_scaled": float(training_manager.weight_decay_scaled),
        },
        "data_loader": train_loader.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
        },
        "yarn_attn_scale": float(model_for_state.yarn.attn_scale),
    }

    model_path, optim_path, meta_path, latest_path = _ckpt_paths(ckpt_dir, step, rank)
    _atomic_torch_save(shard, optim_path)

    # rank0 saves model weights + meta
    if master_process:
        _atomic_torch_save(model_for_state.state_dict(), model_path)
        meta = {
            "version": 1,
            "step": int(step),
            "world_size": int(world_size),
            "run_id": str(run_id),
            "train_files": str(args.train_files),
            "val_files": str(args.val_files),
        }
        _atomic_write_json(meta, meta_path)

    # ensure the checkpoint set is complete before updating latest.txt
    dist.barrier()
    if master_process:
        _atomic_write_text(str(int(step)), latest_path)
    dist.barrier()


def load_checkpoint(*, model: nn.Module, training_manager: TrainingManager, train_loader: ResumableDistributedDataLoader) -> int:
    """Load a resumable checkpoint. Returns the step to resume from (next step index)."""
    ckpt_dir = args.checkpoint_dir
    step = int(args.resume_step) if int(args.resume_step) >= 0 else _find_latest_step(ckpt_dir)

    model_path, optim_path, _meta_path, _latest_path = _ckpt_paths(ckpt_dir, step, rank)

    # Save/load state dicts from the underlying module when torch.compile is used.
    model_for_state = model._orig_mod if hasattr(model, "_orig_mod") else model

    # load per-rank shard first (CPU) so we can validate metadata before touching the model
    shard = torch.load(optim_path, map_location="cpu")
    assert int(shard.get("world_size", world_size)) == int(world_size), (
        f"world_size mismatch: ckpt {shard.get('world_size')} vs current {world_size}"
    )
    start_step = int(shard["step"])

    # rank0 loads weights, then we broadcast parameters+buffers to all ranks
    if master_process:
        model_state = torch.load(model_path, map_location="cpu")
        # torch.compile may prepend all keys with `_orig_mod.`; strip if present for robustness
        if isinstance(model_state, dict) and any(k.startswith("_orig_mod.") for k in model_state.keys()):
            model_state = {k.replace("_orig_mod.", "", 1): v for k, v in model_state.items()}
        model_for_state.load_state_dict(model_state, strict=True)

    for p in model_for_state.parameters():
        dist.broadcast(p.detach(), 0)
    for b in model_for_state.buffers():
        dist.broadcast(b.detach(), 0)

    # restore YaRN attention scaling (not a tensor/buffer)
    _attn_scale = torch.tensor(
        [float(shard.get("yarn_attn_scale", model_for_state.yarn.attn_scale))],
        device=device,
        dtype=torch.float32,
    )
    dist.broadcast(_attn_scale, 0)
    model_for_state.yarn.attn_scale = float(_attn_scale.item())

    # restore optimizer states (move tensors to the local CUDA device)
    opt_states = shard["optimizers"]
    for opt, opt_state in zip(training_manager.optimizers, opt_states):
        opt.load_state_dict(_to_device(opt_state, device))

    # restore TrainingManager schedule state (critical to avoid re-applying YaRN)
    tm = shard["training_manager"]
    training_manager.ws_short = int(tm["ws_short"])
    training_manager.ws_long = int(tm["ws_long"])
    training_manager.batch_size = int(tm["batch_size"])
    training_manager.grad_accum_steps = int(tm["grad_accum_steps"])
    training_manager.batch_lr_scale = float(tm["batch_lr_scale"])
    training_manager.weight_decay_scaled = float(tm.get("weight_decay_scaled", training_manager.weight_decay_scaled))
    training_manager.train_loader_send_args = None

    # YaRN キャッシュを checkpoint の inv_freq と同期（過去 ckpt の pos/pitch 未更新も補正）
    model_for_state.yarn.apply(training_manager.ws_long, training_manager.ws_long)

    # restore training dataloader position
    train_loader.load_state_dict(shard["data_loader"])

    # restore RNG state
    rng = shard.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        torch.cuda.set_rng_state(rng["cuda"])
    if "python" in rng:
        random.setstate(rng["python"])

    dist.barrier()
    if master_process:
        print0(f"Resumed from checkpoint step={step:06d} (next_step={start_step})", console=True)
    dist.barrier()
    return start_step

########################################
#        Training and validation       #
########################################
# A resumable version of the training dataloader (supports .send(), state_dict(), load_state_dict())
train_loader = ResumableDistributedDataLoader(
    args.train_files,
    args.train_bs_schedule[0],
    args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps,
    align_to_bos=True,
)

gc.collect()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations

# Optional resume (must happen after model, optimizers, and train_loader have been created)
start_step = 0
if args.resume:
    start_step = load_checkpoint(model=model, training_manager=training_manager, train_loader=train_loader)
for step in range(start_step, train_steps + 1):
    last_step = (step == train_steps)
    training_manager.advance_schedule(step)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            training_manager.apply_final_ws_ext()
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batches = math.ceil(args.val_tokens / args.val_batch_size)
        val_tokens_effective = val_batches * args.val_batch_size
        if val_tokens_effective != args.val_tokens:
            print0(
                f"val_tokens rounded up from {args.val_tokens:,} to {val_tokens_effective:,} "
                f"to fit val_batch_size {args.val_batch_size:,}",
                console=True,
            )
        val_steps = val_batches * val_grad_accum_steps
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=val_grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens, max_seqlen = next(val_loader)
                with autocast_ctx:
                    val_loss += model(inputs, targets, cum_seqlens, training_manager.get_forward_args(), max_seqlen=max_seqlen)
        val_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for idx in range(training_manager.grad_accum_steps):
        send_args = training_manager.train_loader_send_args
        inputs, targets, cum_seqlens, max_seqlen = train_loader.send(send_args)
        with autocast_ctx:
            (model(inputs, targets, cum_seqlens, training_manager.get_forward_args(), max_seqlen=max_seqlen) / training_manager.grad_accum_steps).backward()
    training_manager.step_optimizers(step)

    # Save a resumable checkpoint at the step boundary (next step index)
    if args.save_checkpoint:
        ckpt_step = step + 1
        if ckpt_step == train_steps or (args.save_every > 0 and ckpt_step % args.save_every == 0):
            save_checkpoint(ckpt_step, model=model, training_manager=training_manager, train_loader=train_loader)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
