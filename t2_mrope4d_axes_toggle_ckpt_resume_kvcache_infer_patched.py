import os
import sys

with open(sys.argv[0]) as f:
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

if hasattr(dynamo.config, "recompile_limit"):
    dynamo.config.recompile_limit = 64


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

# YaRN + 4-axis MRoPE (index / bar / pos / pitch) @classiclarryd
#
# * Interleaved-MRoPE compatible (fixed stride overwrite, default stride=4)
# * half-truncate RoPE is NOT used here (full rotary frequencies)


def forward_fill_from_updates(updates: Tensor, update_mask: Tensor, idx: Tensor) -> Tensor:
    """Forward-fill (last observation carried forward).

    本コードの用途（MRoPE の bar/pos/pitch インデックス生成）に特化して、
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
    """YaRN + 4-axis MRoPE（index / bar / pos / pitch）.

    互換性目標:
      * Qwen 系で使われる Interleaved-MRoPE の「固定ストライド上書き」を 4軸に拡張。

    設計方針:
      * trig（cos/sin）は forward では計算しない（キャッシュ参照のみ）
      * 4軸の cos/sin をトークン列から生成し、各層で共有して使用
      * torch.compile に乗る（Python ループなし）

    注意:
      * half-truncate RoPE は使用しない（全周波数で回転する）
      * 4軸は enable_index/enable_bar/enable_pos/enable_pitch で個別に ON/OFF 可能
        - OFF の軸は「上書きしない」ため、その軸に割り当てられたペアは base（index または identity）で回転する
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: int = 10000,
        *,
        # token ID 設定（本コードベースのデフォルト語彙に合わせる）
        pitch_start: int = 0,
        pitch_size: int = 96,
        bar_id: int | None = 96,
        pos_start: int = 97,
        pos_size: int = 32,
        # doc 境界（BOS=129 が既定。None で無効化）
        doc_id: int | None = 129,
        # pitch/pos の forward-fill と reset 条件
        carry_pitch: bool = True,
        reset_pitch_on_pos: bool = True,
        reset_pitch_on_bar: bool = True,
        # 0 を「軸無効」に予約したい場合のオフセット
        pos_idx_offset: int = 1,
        pitch_idx_offset: int = 1,
        # pitch 回転のゲート
        pitch_gate: str = 'pitch_tok_only',
        # rotary pairs の割当（head_dim/2 を 4軸に分配）
        axial_fractions: tuple[int, int, int, int] = (1, 1, 1, 1),
        # ペア並び: 'interleave'（Qwen互換） or 'block'
        pair_layout: str = 'interleave',
        # 軸ごとの ON/OFF（学習中に切り替える場合は torch.compile の再コンパイル要因になり得る）
        enable_index: bool = True,
        enable_bar: bool = True,
        enable_pos: bool = True,
        enable_pitch: bool = True,
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

        self.pitch_gate = str(pitch_gate).lower()
        if self.pitch_gate not in ('none', 'pitch_tok_only'):
            raise ValueError(f"pitch_gate must be 'none' or 'pitch_tok_only', got {pitch_gate!r}")

        self.pair_layout = str(pair_layout).lower()
        if self.pair_layout not in ('block', 'interleave'):
            raise ValueError(f"pair_layout must be 'block' or 'interleave', got {pair_layout!r}")

        # axis enables (kept as Python bools: no tensor overhead)
        self.enable_index = bool(enable_index)
        self.enable_bar = bool(enable_bar)
        self.enable_pos = bool(enable_pos)
        self.enable_pitch = bool(enable_pitch)

        # split head_dim/2 pairs across 4 axes
        total_pairs = self.head_dim // 2
        self.pairs_index, self.pairs_bar, self.pairs_pos, self.pairs_pitch = self._allocate_pairs(
            total_pairs, axial_fractions
        )
        self.mrope_section = (self.pairs_index, self.pairs_bar, self.pairs_pos, self.pairs_pitch)
        if sum(self.mrope_section) != total_pairs:
            raise RuntimeError('Internal error: mrope_section does not sum to total_pairs')

        # Validate interleaved fixed-stride feasibility (enabled axes only)
        if self.pair_layout == 'interleave':
            num_axes = 4
            for axis_id, sec in enumerate(self.mrope_section):
                if axis_id == 0 or sec <= 0:
                    continue
                if axis_id == 1 and (not self.enable_bar):
                    continue
                if axis_id == 2 and (not self.enable_pos):
                    continue
                if axis_id == 3 and (not self.enable_pitch):
                    continue
                max_idx = axis_id + num_axes * (sec - 1)
                if max_idx >= total_pairs:
                    raise ValueError(
                        "pair_layout='interleave' cannot represent this allocation with fixed stride: "
                        f"axis={axis_id} sec={sec} total_pairs={total_pairs} -> max_idx={max_idx}. "
                        "Reduce axial_fractions for non-base axes or use pair_layout='block'."
                    )

        self.reset()

    @staticmethod
    def _allocate_pairs(total_pairs: int, ratios: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r0, r1, r2, r3 = (int(r) for r in ratios)
        if min(r0, r1, r2, r3) <= 0:
            raise ValueError(f"axial_fractions must be positive, got {ratios}")
        denom = r0 + r1 + r2 + r3
        scaled = [total_pairs * r0, total_pairs * r1, total_pairs * r2, total_pairs * r3]
        base = [s // denom for s in scaled]
        frac = [s % denom for s in scaled]
        rem = total_pairs - sum(base)
        order = sorted(range(4), key=lambda i: (-frac[i], i))
        for i in order[:rem]:
            base[i] += 1
        return base[0], base[1], base[2], base[3]

    def set_axes(
        self,
        *,
        index: bool | None = None,
        bar: bool | None = None,
        pos: bool | None = None,
        pitch: bool | None = None,
    ) -> None:
        """4軸の有効/無効を切り替える（必要なら外部から呼ぶ）."""
        if index is not None:
            self.enable_index = bool(index)
        if bar is not None:
            self.enable_bar = bool(bar)
        if pos is not None:
            self.enable_pos = bool(pos)
        if pitch is not None:
            self.enable_pitch = bool(pitch)

    def reset(self):
        # base inv_freq (full RoPE; no half-truncate)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        # inv_freq participates in YaRN updates and must be checkpointed for exact resume.
        self.inv_freq = nn.Buffer(inv_freq, persistent=True)

        # cos/sin caches per axis (all axes share the same inv_freq; axis ごとに position id だけを変える)
        total_pairs = self.head_dim // 2

        # index: [0..T)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, inv_freq)
        self._cos_index = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=True)
        self._sin_index = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=True)
        assert self._cos_index.size(1) == total_pairs

        # bar: cumulative count, upper bound <= seq_len
        t_bar = torch.arange(self.max_seq_len + 1, dtype=torch.float32, device=device)
        theta_bar = torch.outer(t_bar, inv_freq)
        self._cos_bar = nn.Buffer(theta_bar.cos().to(torch.bfloat16), persistent=True)
        self._sin_bar = nn.Buffer(theta_bar.sin().to(torch.bfloat16), persistent=True)

        # pos/pitch: small categorical indices
        t_pos = torch.arange(self.pos_cache_len, dtype=torch.float32, device=device)
        theta_pos = torch.outer(t_pos, inv_freq)
        self._cos_pos = nn.Buffer(theta_pos.cos().to(torch.bfloat16), persistent=True)
        self._sin_pos = nn.Buffer(theta_pos.sin().to(torch.bfloat16), persistent=True)

        t_pitch = torch.arange(self.pitch_cache_len, dtype=torch.float32, device=device)
        theta_pitch = torch.outer(t_pitch, inv_freq)
        self._cos_pitch = nn.Buffer(theta_pitch.cos().to(torch.bfloat16), persistent=True)
        self._sin_pitch = nn.Buffer(theta_pitch.sin().to(torch.bfloat16), persistent=True)

        # (T,) 用の 0..max_seq_len-1 インデックス（forward で arange を作らない）
        self._idx_cache = nn.Buffer(
            torch.arange(self.max_seq_len, dtype=torch.int32, device=device), persistent=False
        )

        # attn scale
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        # YaRN frequency scaling (index/bar 用の inv_freq を更新し、キャッシュも更新)
        rotations = args.block_size * old_window * self.inv_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.inv_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)

        # rebuild index cache
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        theta = torch.outer(t, self.inv_freq)
        self._cos_index.copy_(theta.cos().to(self._cos_index.dtype))
        self._sin_index.copy_(theta.sin().to(self._sin_index.dtype))

        # rebuild bar cache
        t_bar = torch.arange(self.max_seq_len + 1, dtype=torch.float32, device=self.inv_freq.device)
        theta_bar = torch.outer(t_bar, self.inv_freq)
        self._cos_bar.copy_(theta_bar.cos().to(self._cos_bar.dtype))
        self._sin_bar.copy_(theta_bar.sin().to(self._sin_bar.dtype))

        # pos/pitch は短距離で使うことが多いため、ここでは変更しない（必要なら reset() で全更新）
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
        use_bar = self.enable_bar and (self.pairs_bar > 0)
        use_pos = self.enable_pos and (self.pairs_pos > 0)
        use_pitch = self.enable_pitch and (self.pairs_pitch > 0)
        need_extra = use_bar or use_pos or use_pitch

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
        cos_bar = sin_bar = None
        if use_bar:
            # bar index: cumulative count of bar tokens, reset at doc boundaries
            bar_idx = torch.cumsum(bar_tok.to(dtype=torch.int32), dim=0, dtype=torch.int32)
            if self.doc_id is not None:
                bar_offset = forward_fill_from_updates(bar_idx, doc_tok, idx)
                bar_idx = bar_idx - bar_offset
            bar_idx = bar_idx.clamp(0, self._cos_bar.size(0) - 1)
            cos_bar = F.embedding(bar_idx, self._cos_bar)
            sin_bar = F.embedding(bar_idx, self._sin_bar)

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
        if use_pitch:
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
            s0, s1, s2, s3 = self.mrope_section
            o1 = s0
            o2 = s0 + s1
            o3 = s0 + s1 + s2
            if use_bar:
                cos[:, o1:o2] = cos_bar[:, o1:o2]
                sin[:, o1:o2] = sin_bar[:, o1:o2]
            if use_pos:
                cos[:, o2:o3] = cos_pos[:, o2:o3]
                sin[:, o2:o3] = sin_pos[:, o2:o3]
            if use_pitch:
                cos[:, o3:] = cos_pitch[:, o3:]
                sin[:, o3:] = sin_pitch[:, o3:]
        else:
            # interleaved fixed-stride overwrite (Qwen style)
            stride = 4
            if use_bar:
                end = int(self.pairs_bar) * stride
                cos[:, 1:end:stride] = cos_bar[:, 1:end:stride]
                sin[:, 1:end:stride] = sin_bar[:, 1:end:stride]
            if use_pos:
                end = int(self.pairs_pos) * stride
                cos[:, 2:end:stride] = cos_pos[:, 2:end:stride]
                sin[:, 2:end:stride] = sin_pos[:, 2:end:stride]
            if use_pitch:
                end = int(self.pairs_pitch) * stride
                cos[:, 3:end:stride] = cos_pitch[:, 3:end:stride]
                sin[:, 3:end:stride] = sin_pitch[:, 3:end:stride]

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
    seqlens: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    window_size: tuple[int, int]


from flash_attn import flash_attn_interface


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


    def forward(self, x: Tensor, ve: Tensor, attn_args: AttnArgs):
        B, T, _ = x.size()
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos = attn_args.cos[:, :T]
        sin = attn_args.sin[:, :T]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        max_len = args.train_max_seq_len if self.training else args.device_batch_size_tokens
        y = flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=attn_args.seqlens, cu_seqlens_k=attn_args.seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=True, softmax_scale=attn_args.attn_scale, window_size=attn_args.window_size
        )
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

    def forward(self, x: Tensor, ve: Tensor, attn_args: AttnArgs):
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

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

        B = 1
        T = input_seq.size(0)
        assert T <= self.yarn.max_seq_len, "sequence length exceeds rotary cache"

        x = self.transformer.wte(input_seq)
        x = norm(x)[None]
        x0 = x

        bm_sizes = self._get_bm_sizes(ws_short, ws_long)

        # YaRN + 4-axis MRoPE: generate cos/sin once per sequence and share across layers
        cos, sin = self.yarn(input_seq)

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_seq) if str(i) in self.value_embeds else None
            attn_args = AttnArgs(
                seqlens=seqlens,
                cos=cos,
                sin=sin,
                attn_scale=self.yarn.attn_scale,
                window_size=(bm_sizes[i], 0),
            )
            x = block(x, ve, attn_args)
        x = norm(x)

        softcap = 15.0

        if not self.training:
            loss = 0.0
            x_flat = x.flatten(end_dim=1)
            x_chunks = x_flat.chunk(4)
            t_chunks = target_seq.chunk(4)
            for x_chunk, t_chunk in zip(x_chunks, t_chunks):
                logits = F.linear(x_chunk, self.lm_head.weight).float()
                logits = logits[:, :self.vocab_size]
                logits = softcap * torch.tanh(logits / softcap)
                loss += F.cross_entropy(logits, t_chunk, reduction="mean") / 4
            return loss

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

BOS_ID = 129

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
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True)
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
      `(inputs, targets, cum_seqlens)`.
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
            max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

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

                buf = torch.cat([self.tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
                _inputs = buf[:-1]
                _targets = buf[1:]
                end_idxs[-1] -= 1
                cum_lengths = (end_idxs - start_idxs).cumsum(0)

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
                self.pos += self._num_tokens

            _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
            _cum_lengths[0] = 0
            _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

            _inputs = _inputs.to(dtype=torch.int32)
            _targets = _targets.to(dtype=torch.int64)
            _cum_lengths = _cum_lengths.to(dtype=torch.int32)

            return (
                _inputs.to(device="cuda", non_blocking=True),
                _targets.to(device="cuda", non_blocking=True),
                _cum_lengths.to(device="cuda", non_blocking=True),
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
    train_files: str = "train.bin" # input .bin to train on
    val_files: str = "val.bin" # input .bin to eval validation loss on
    val_tokens: int = 32 * 2048 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    train_bs_schedule: tuple = (4 * 2048, 4 * 2048, 8 * 2048, 8 * 2048, 
                                32 * 2048, 32 * 2048, 32 * 2048, 32 * 2048,
                                32 * 2048, 32 * 2048, 32 * 2048, 32 * 2048
                               )
    train_bs_extension: int = 32 * 2048
    train_max_seq_len: int = 128 * 16 * 2 # doubled to enable longer window sizes
    val_batch_size: int = 32 * 2048
    device_batch_size_tokens: int = train_max_seq_len  # per-rank sequence length (varlen B==1)
    reference_batch_size: int = 2**19
    # optimization
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.3
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    num_scheduled_iterations: int = 4700  # number of steps to complete ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
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
    checkpoint_dir: str = ""  # "" -> logs/{run_id}
    resume: bool = False
    resume_step: int = -1  # -1 -> latest
    # attention masking
    block_size: int = 128
    window_pattern: str = "SSSL"
    ws_schedule: tuple = (1, 7, 11, 15,
                          19, 23, 23, 23,
                          23, 23, 23, 23)
    ws_final: int = 23 # set final validation ws, used for YaRN extension and short window size
    ws_validate_post_yarn_ext: int = 27 # extend long windows out even further after applying YaRN
    # model (GQA) - 0 means use num_heads (GQA disabled, nanochat default)
    num_kv_heads: int = 0

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)
args.total_batch_size_tokens = args.train_bs_schedule[0]

# -----------------------------------------------------------------------------
# Inference utilities (KV-cache toggle, MIDI <-> token, FlashAttention)
# -----------------------------------------------------------------------------
from typing import Optional, Sequence, Tuple
import inspect


class KVCache:
    """Per-layer KV cache for FlashAttention kv-cache inference.

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
        # FlashAttention-3 may propagate NaN/inf from uninitialized KV-cache padding into the output.
        # Initializing once at allocation time keeps padding finite without per-step overhead.
        k.zero_()
        v.zero_()
        cache_seqlens = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        return KVCache(k, v, cache_seqlens)

    def reset(self):
        self.cache_seqlens.zero_()

    def advance(self, n: int):
        # cache_seqlens is int32; keep it there to match flash-attn kernels.
        self.cache_seqlens += int(n)

    def layer(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[i], self.v_cache[i]


class YarnState:
    """Incremental YaRN + 4-axis MRoPE state for O(1) per-token rotary in decode.

    This avoids the O(T) re-evaluation cost of Yarn.forward() at every decode step,
    preserving the speed benefits of KV-cache decoding.
    """

    def __init__(self, yarn: "Yarn", *, device: torch.device, dtype: torch.dtype):
        self.yarn = yarn
        self.device = device
        self.dtype = dtype

        # Snapshot caches in the inference dtype to avoid fp16/bf16 mixing (-> fp32 upcast).
        # The caches are small (<= max_seq_len x head_dim/2), so this is negligible overhead.
        y = yarn
        self._cos_index = y._cos_index.to(device=device, dtype=dtype)
        self._sin_index = y._sin_index.to(device=device, dtype=dtype)
        self._cos_bar = y._cos_bar.to(device=device, dtype=dtype)
        self._sin_bar = y._sin_bar.to(device=device, dtype=dtype)
        self._cos_pos = y._cos_pos.to(device=device, dtype=dtype)
        self._sin_pos = y._sin_pos.to(device=device, dtype=dtype)
        self._cos_pitch = y._cos_pitch.to(device=device, dtype=dtype)
        self._sin_pitch = y._sin_pitch.to(device=device, dtype=dtype)

        self.reset()

    def reset(self):
        self.idx_pos = 0  # absolute token index (for index axis)
        self.bar_count = 0
        self.pos_state = 0
        self.pitch_state = 0

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
        bar_tok = tok == int(y.bar_id)
        pos_tok = (tok >= int(y.pos_start)) & (tok < int(y.pos_start + y.pos_size))
        doc_tok = tok == int(y.doc_id) if y.doc_id is not None else torch.zeros_like(tok, dtype=torch.bool)

        # bar axis state
        bar_idx = torch.cumsum(bar_tok.to(torch.int32), dim=0)
        if y.doc_id is not None:
            bar_offset = forward_fill_from_updates(bar_idx, doc_tok, idx)
            bar_idx = bar_idx - bar_offset
        self.bar_count = int(bar_idx[-1].item())

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

        # token type flags (model vocab)
        pitch_tok = (tok >= int(y.pitch_start)) and (tok < int(y.pitch_start + y.pitch_size))
        bar_tok = tok == int(y.bar_id)
        pos_tok = (tok >= int(y.pos_start)) and (tok < int(y.pos_start + y.pos_size))
        doc_tok = (tok == int(y.doc_id)) if (y.doc_id is not None) else False

        # Update bar axis state (cumsum semantics: bar token increments at itself)
        if doc_tok:
            self.bar_count = 0
        elif bar_tok:
            self.bar_count += 1
        bar_idx = self.bar_count
        if bar_idx < 0:
            bar_idx = 0
        if bar_idx >= self._cos_bar.size(0):
            bar_idx = self._cos_bar.size(0) - 1

        # Update pos axis state (forward-fill with resets on bar/doc)
        if doc_tok or bar_tok:
            self.pos_state = 0
        elif pos_tok:
            self.pos_state = tok - int(y.pos_start) + int(y.pos_idx_offset)
        pos_idx = self.pos_state
        if pos_idx < 0:
            pos_idx = 0
        if pos_idx >= self._cos_pos.size(0):
            pos_idx = self._cos_pos.size(0) - 1

        # Update pitch axis state (pre-gate)
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
        pitch_state = self.pitch_state

        # apply pitch gate (output index only)
        if getattr(y, "pitch_gate", "pitch_tok_only") == "pitch_tok_only" and (not pitch_tok):
            pitch_idx = 0
        else:
            pitch_idx = pitch_state
        if pitch_idx < 0:
            pitch_idx = 0
        if pitch_idx >= self._cos_pitch.size(0):
            pitch_idx = self._cos_pitch.size(0) - 1

        # base index cos/sin (index axis can be disabled -> identity row 0)
        base_t = t if bool(y.enable_index) else 0
        if base_t >= self._cos_index.size(0):
            raise RuntimeError(f"Sequence length {t} exceeds YaRN cache (max_seq_len={self._cos_index.size(0)})")
        cos = self._cos_index[base_t].clone()
        sin = self._sin_index[base_t].clone()

        # overwrite axial sections (same logic as Yarn.forward)
        stride = 4
        s0, s1, s2, s3 = y.mrope_section

        # BAR axis overwrite
        if bool(y.enable_bar) and s1 > 0:
            cos_bar = self._cos_bar[bar_idx]
            sin_bar = self._sin_bar[bar_idx]
            if y.pair_layout == "block":
                o1 = s0
                o2 = s0 + s1
                cos[o1:o2] = cos_bar[o1:o2]
                sin[o1:o2] = sin_bar[o1:o2]
            else:  # interleave
                end = s1 * stride
                cos[1:end:stride] = cos_bar[1:end:stride]
                sin[1:end:stride] = sin_bar[1:end:stride]

        # POS axis overwrite
        if bool(y.enable_pos) and s2 > 0:
            cos_pos = self._cos_pos[pos_idx]
            sin_pos = self._sin_pos[pos_idx]
            if y.pair_layout == "block":
                o2 = s0 + s1
                o3 = s0 + s1 + s2
                cos[o2:o3] = cos_pos[o2:o3]
                sin[o2:o3] = sin_pos[o2:o3]
            else:
                end = s2 * stride
                cos[2:end:stride] = cos_pos[2:end:stride]
                sin[2:end:stride] = sin_pos[2:end:stride]

        # PITCH axis overwrite
        if bool(y.enable_pitch) and s3 > 0:
            cos_pitch = self._cos_pitch[pitch_idx]
            sin_pitch = self._sin_pitch[pitch_idx]
            if y.pair_layout == "block":
                o3 = s0 + s1 + s2
                cos[o3:] = cos_pitch[o3:]
                sin[o3:] = sin_pitch[o3:]
            else:
                end = s3 * stride
                cos[3:end:stride] = cos_pitch[3:end:stride]
                sin[3:end:stride] = sin_pitch[3:end:stride]

        self.idx_pos += 1

        cos = cos.view(1, 1, 1, -1)
        sin = sin.view(1, 1, 1, -1)
        return cos, sin


# ---- FlashAttention compatibility wrappers (minimize Python overhead) ----

_KW_CACHE: dict[int, set[str]] = {}


def _get_sig_params(func) -> set[str]:
    fid = id(func)
    params = _KW_CACHE.get(fid)
    if params is None:
        params = set(inspect.signature(func).parameters.keys())
        _KW_CACHE[fid] = params
    return params


def _maybe_add_window_size_lr(params: set[str], call_kwargs: dict, window_size):
    """Some FlashAttention builds expose window_size_left/right instead of window_size."""
    if window_size is None:
        return
    if ("window_size" not in params) and ("window_size_left" in params or "window_size_right" in params):
        # Only add if caller didn't already provide them.
        if "window_size_left" in params and "window_size_left" not in call_kwargs:
            call_kwargs["window_size_left"] = int(window_size[0])
        if "window_size_right" in params and "window_size_right" not in call_kwargs:
            call_kwargs["window_size_right"] = int(window_size[1])


def _flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    fn = getattr(flash_attn_interface, "flash_attn_func", None)
    if fn is None:
        raise RuntimeError("flash_attn_interface.flash_attn_func is not available in this FlashAttention build.")
    if not kwargs:
        return fn(q, k, v)
    params = _get_sig_params(fn)
    call_kwargs = {}
    for kk, vv in kwargs.items():
        if kk in params:
            call_kwargs[kk] = vv
    # window_size tuple -> window_size_left/right fallback
    if "window_size" in kwargs:
        _maybe_add_window_size_lr(params, call_kwargs, kwargs.get("window_size"))
    return fn(q, k, v, **call_kwargs)


def _flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    fn = getattr(flash_attn_interface, "flash_attn_with_kvcache", None)
    if fn is None:
        raise RuntimeError("flash_attn_interface.flash_attn_with_kvcache is not available in this FlashAttention build.")

    params = _get_sig_params(fn)
    call_kwargs = {}

    # Required / common args
    if "k" in params:
        call_kwargs["k"] = k
    if "v" in params:
        call_kwargs["v"] = v
    if "cache_seqlens" in params:
        call_kwargs["cache_seqlens"] = cache_seqlens
    if "cache_seqlens_q" in params:
        call_kwargs["cache_seqlens_q"] = cache_seqlens
    if "cache_seqlens_k" in params:
        call_kwargs["cache_seqlens_k"] = cache_seqlens

    # Pass through supported kwargs (causal, softmax_scale, window_size, etc.)
    for kk, vv in kwargs.items():
        if kk in params:
            call_kwargs[kk] = vv

    # window_size tuple -> window_size_left/right fallback
    if "window_size" in kwargs:
        _maybe_add_window_size_lr(params, call_kwargs, kwargs.get("window_size"))

    return fn(q, k_cache, v_cache, **call_kwargs)


# ---- Inference forward path (no changes to training forward) ----

@torch.no_grad()
def _attn_forward_infer(
    attn: CausalSelfAttention,
    x: torch.Tensor,
    ve: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attn_scale: float,
    window_size: Tuple[int, int],
    use_kv_cache: bool,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
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

    if use_kv_cache:
        assert k_cache is not None and v_cache is not None and cache_seqlens is not None
        y = _flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            softmax_scale=attn_scale,
            window_size=window_size,
        )
    else:
        y = _flash_attn_func(
            q,
            k,
            v,
            causal=True,
            softmax_scale=attn_scale,
            window_size=window_size,
        )

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

    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](input_seq) if str(i) in model.value_embeds else None

        if kv_cache is not None and use_kv_cache:
            k_cache_i, v_cache_i = kv_cache.layer(i)
            cache_seqlens = kv_cache.cache_seqlens
        else:
            k_cache_i = v_cache_i = cache_seqlens = None

        # attention
        x = x + _attn_forward_infer(
            block.attn,
            norm(x),
            ve,
            cos[:, :T],
            sin[:, :T],
            attn_scale=model.yarn.attn_scale,
            window_size=(bm_sizes[i], 0),
            use_kv_cache=(kv_cache is not None and use_kv_cache),
            k_cache=k_cache_i,
            v_cache=v_cache_i,
            cache_seqlens=cache_seqlens,
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


def _ensure_bos_bar_pos0(tokens: Sequence[int], *, bos_id: int, bar_id: int, pos0_id: int) -> list[int]:
    out = list(tokens)
    if len(out) == 0 or out[0] != bos_id:
        out = [bos_id] + out
    # ensure next is BAR
    if len(out) == 1 or out[1] != bar_id:
        out.insert(1, bar_id)
    # ensure next is POS0
    if len(out) == 2 or out[2] != pos0_id:
        out.insert(2, pos0_id)
    return out


@torch.no_grad()
def _sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    allowed_mask: Optional[torch.Tensor],
) -> int:
    # logits_1d: [vocab]
    if allowed_mask is not None:
        logits_1d = logits_1d.masked_fill(~allowed_mask, float("-inf"))
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits_1d).item())

    x = logits_1d / float(temperature)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), x.size(-1))
        v, _ = torch.topk(x, k)
        cutoff = v[-1]
        x = torch.where(x < cutoff, float("-inf"), x)

    probs = torch.softmax(x, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())


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
    out = []
    for t in tokens_model:
        ti = int(t)
        if ti >= 128:  # BAR and above in vocab164 mapping
            ti = ti - int(shift)
        out.append(ti)
    return out


def _remi96_ids_to_model(tokens_remi: Sequence[int], *, shift: int) -> list[int]:
    out = []
    for t in tokens_remi:
        ti = int(t)
        if ti >= 96:  # BAR and above in REMI96 mapping
            ti = ti + int(shift)
        out.append(ti)
    return out


def _remi96_tokens_to_midi(
    tokens_remi: Sequence[int],
    *,
    out_path: str,
    meta: dict,
    velocity: int = 100,
    duration: int = 1,
):
    """Decode REMI96 tokens (BOS/BAR/POS/PITCH/EOF) to a MIDI file.

    This mirrors tensor_to_midi_remi96.py to minimize format drift.
    """
    import numpy as np
    from symusic import Note, Score, Track

    PITCH_MIN = 0
    PITCH_MAX = 95
    BAR_ID = 96
    POS_OFFSET = 97

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
    current_pos: Optional[int] = None
    eof_seen = False

    while idx < len(tokens):
        tok = int(tokens[idx])
        idx += 1
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
            # ignore notes before first BAR/POS
            continue
        abs_time = current_bar * bar_ticks + current_pos
        times_list.append(abs_time)
        pitches_list.append(tok)

    if not eof_seen:
        # allow decoding without explicit EOF
        pass

    if len(times_list) == 0:
        # empty -> create an empty score
        score = Score(tpq=int(tpq))
        score.tracks = [Track(name="track0")]
        score.dump_midi(out_path)
        return

    times = np.asarray(times_list, dtype=np.int32)
    pitches = np.asarray(pitches_list, dtype=np.int32)

    # undo time_shift (reltime -> absolute ticks)
    if int(time_shift) != 0:
        times = times - int(time_shift)

    # convert to ticks
    ticks = (times.astype(np.int64) * int(tpq)) // 1

    notes = [Note(int(t), int(duration), int(p), int(velocity)) for t, p in zip(ticks.tolist(), pitches.tolist())]
    tr = Track(name="track0")
    tr.notes = notes
    score = Score(tpq=int(tpq))
    score.tracks = [tr]
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

    parser = argparse.ArgumentParser(description="KV-cache inference (FlashAttention) + MIDI I/O")
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
    parser.add_argument("--tpq", type=int, default=8)
    parser.add_argument("--min_dur", type=int, default=1)
    parser.add_argument("--include_drums", type=int, default=0)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--remi_shift", type=int, default=32, help="REMI96->model vocab shift for BAR/POS/BOS/EOF/PAD (default 32).")
    parser.add_argument("--save_tokens", type=str, default="", help="Optional: save generated model-token ids to .npy")
    parser.add_argument("--save_meta", type=str, default="", help="Optional: save meta json (includes tokenizer meta + generation args).")
    parser.add_argument("--ws_short", type=int, default=-1, help="Short window size (in blocks). Default: from ckpt (if available) else args.ws_final")
    parser.add_argument("--ws_long", type=int, default=-1, help="Long window size (in blocks). Default: from ckpt (if available) else args.ws_final")
    parser.add_argument("--eof_id", type=int, default=162, help="EOF token id in model vocab (default 162).")
    parser.add_argument("--mask_unused_pitches", type=int, default=1, help="Mask model tokens 96..127 (REMI96 unused pitches) during sampling.")
    parser.add_argument("--disallow_bos_pad", type=int, default=1, help="Disallow BOS/PAD during sampling (recommended).")

    a = parser.parse_args()
    if not a.infer:
        raise RuntimeError("--infer flag missing (internal).")

    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    device = torch.device(a.device)
    if device.type != "cuda":
        raise RuntimeError("FlashAttention inference requires a CUDA device.")
    torch.cuda.set_device(device)
    globals()["device"] = device  # Yarn.reset() uses global 'device' in this codebase

    # Dtype selection (FlashAttention kernels support bf16/fp16)
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
    if bar_ticks != 32:
        raise RuntimeError(
            f"bar_ticks_reltime={bar_ticks} (expected 32). "
            "このモデルは pos_size=32 のREMI表現を前提にしています。--tpq 8 (4/4) で再トークナイズしてください。"
        )

    tokens_model = _remi96_ids_to_model(tokens_remi.tolist(), shift=int(a.remi_shift))

    # Range selection
    sl = _parse_slice(a.prompt_range, len(tokens_model))
    tokens_model = tokens_model[sl]

    # Ensure minimal valid prefix for decoding & YaRN resets
    BOS_ID = 129
    BAR_ID_MODEL = 96
    POS0_ID_MODEL = 97 # pos=0 with offset=129
    tokens_model = _ensure_bos_bar_pos0(tokens_model, bos_id=BOS_ID, bar_id=BAR_ID_MODEL, pos0_id=POS0_ID_MODEL)

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
    vocab_size = 132
    head_dim = 128  # must match training; used for RoPE + attention reshape

    # Derive num_layers from checkpoint if possible (robust to config tweaks)
    num_layers = 16
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
    model_dim = 1024
    num_heads = 8
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

    # Build allowed mask
    allowed_mask = None
    if int(a.mask_unused_pitches) == 1:
        allowed = torch.zeros((vocab_size,), device=device, dtype=torch.bool)
        allowed[:96] = True                      # pitch 0..95
        allowed[128:vocab_size] = True           # BAR/POS/BOS/EOF/PAD
        if int(a.disallow_bos_pad) == 1:
            allowed[BOS_ID] = False
            allowed[131] = False  # PAD
        allowed_mask = allowed
    elif int(a.disallow_bos_pad) == 1:
        allowed = torch.ones((vocab_size,), device=device, dtype=torch.bool)
        allowed[BOS_ID] = False
        allowed[163] = False
        allowed_mask = allowed

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
            next_id = _sample_next_token(
                last_logits,
                temperature=float(a.temperature),
                top_k=int(a.top_k),
                allowed_mask=allowed_mask,
            )
            generated.append(int(next_id))
            if int(next_id) == eof_id:
                break
            if len(generated) >= max_total:
                break

            cos1, sin1 = yarn_state.step(int(next_id))
            tok1 = torch.tensor([int(next_id)], device=device, dtype=torch.long)
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
        # Full recompute mode (no KV cache): still uses FlashAttention kernels.
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
            next_id = _sample_next_token(
                last_logits,
                temperature=float(a.temperature),
                top_k=int(a.top_k),
                allowed_mask=allowed_mask,
            )
            generated.append(int(next_id))
            if int(next_id) == eof_id:
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
    "device_batch_size_tokens must be a multiple of 16 (flash_attn varlen requirement)"
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

num_layers = 16
num_heads = 8
head_dim = 128
model_dim = 1024
model: nn.Module = GPT(
    vocab_size=164,
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
                inputs, targets, cum_seqlens = next(val_loader)
                with autocast_ctx:
                    val_loss += model(inputs, targets, cum_seqlens, training_manager.get_forward_args())
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
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        with autocast_ctx:
            (model(inputs, targets, cum_seqlens, training_manager.get_forward_args()) / training_manager.grad_accum_steps).backward()
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
