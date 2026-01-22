import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import math
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

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from torch import Tensor, nn

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

# yarn implementation @classiclarryd (RoPE base from nanochat)
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.reset()

    def reset(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        self.inv_freq = inv_freq
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        self.cos = nn.Buffer(cos[None, :, None, :], persistent=False)
        self.sin = nn.Buffer(sin[None, :, None, :], persistent=False)
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = args.block_size * old_window * self.inv_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.inv_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.cos.copy_(freqs.cos().to(self.cos.dtype)[None, :, None, :])
        self.sin.copy_(freqs.sin().to(self.sin.dtype)[None, :, None, :])
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


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
        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        pattern = self.window_pattern
        window_sizes = []
        for layer_idx in range(self.num_layers):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(long_bm if char == "L" else short_bm)
        window_sizes[-1] = long_bm
        return window_sizes

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long

        B = 1
        T = input_seq.size(0)
        assert T <= self.yarn.cos.size(1), "sequence length exceeds rotary cache"

        x = self.transformer.wte(input_seq)
        x = norm(x)[None]
        x0 = x

        bm_sizes = self._get_bm_sizes(ws_short, ws_long)

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_seq) if str(i) in self.value_embeds else None
            attn_args = AttnArgs(
                seqlens=seqlens,
                cos=self.yarn.cos,
                sin=self.yarn.sin,
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

BOS_ID = 161

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
    return min(11,args.ws_schedule[ws_idx] // 2), args.ws_schedule[ws_idx]

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
        self.ws_long = args.ws_validate_post_yarn_ext

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
        # only apply yarn for first few
        if new_ws_long != self.ws_long and new_ws_long<=13:
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
    train_bs_schedule: tuple = (4 * 2048, 8 * 2048, 16 * 2048, 32 * 2048, 
                                32 * 2048, 32 * 2048, 32 * 2048, 32 * 2048,
                                32 * 2048, 32 * 2048, 32 * 2048, 32 * 2048
                               )
    train_bs_extension: int = 32 * 2048
    train_max_seq_len: int = 128 * 16 * 2 # doubled to enable longer window sizes
    val_batch_size: int = 32 * 2048
    device_batch_size_tokens: int = 1024  # tokens per micro-batch per rank (varlen)
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
    warmdown_ratio: float = 0.4
    final_lr_frac: float = 0.0
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    window_pattern: str = "SSSL"
    ws_schedule: tuple = (3, 7, 11, 13,
                          15, 17, 19, 21,
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

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(args.train_files, args.train_bs_schedule[0], args.train_max_seq_len, grad_accum_steps=grad_accum_steps)

gc.collect()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
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
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = val_grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=val_grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
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
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in training_manager.optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for idx in range(training_manager.grad_accum_steps):
        send_args = training_manager.train_loader_send_args
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (model(inputs, targets, cum_seqlens, training_manager.get_forward_args()) / training_manager.grad_accum_steps).backward()
    training_manager.step_optimizers(step)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
