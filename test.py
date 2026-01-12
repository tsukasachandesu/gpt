from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    """[a0,a1) と [b0,b1) が重なるか。"""
    return (a0 < b1) and (b0 < a1)


class FundamentalMusicEmbedding(nn.Module):
    """
    Guo et al. (AAAI 2023) の Fundamental Music Embedding (FME) を実装。

    論文の定義（d は d_model）:  
      w_k = B^{-2k/d}
      P_k(f) = [ sin(w_k f) + b_sin^k , cos(w_k f) + b_cos^k ]
      FME(f) = [P_0(f), ..., P_{d/2-1}(f)]  (concatenate)

    実装上のポイント:
    - 周波数 w_k は d/2 本だけ持ち、偶数次元に sin、奇数次元に cos を詰める（中間を半分に）。
    - cache_size を与えると、入力が torch.long のときに
      sin/cos 部分をテーブル参照にして高速化（学習可能バイアスは別途加算）。
    """

    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        bias_init: str = "zeros",   # "zeros" or "normal"
        bias_std: float = 0.02,
        cache_size: Optional[int] = None,  # 例: pitch=128, pos=32 など
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairs (got {d_model}).")

        self.d_model = int(d_model)
        self.base = float(base)
        self.cache_size = int(cache_size) if cache_size is not None else None

        # k = 0..d/2-1 に対して w_k = B^{-2k/d}
        k = torch.arange(self.d_model // 2, dtype=torch.float32)
        freqs = torch.pow(torch.tensor(self.base, dtype=torch.float32), (-2.0 * k) / float(self.d_model))
        self.register_buffer("freqs", freqs, persistent=True)

        # sin/cos 部分だけのテーブル（バイアスは含めない：学習で変わるため）
        if self.cache_size is not None:
            if self.cache_size <= 0:
                raise ValueError(f"cache_size must be positive (got {self.cache_size}).")

            values = torch.arange(self.cache_size, dtype=torch.float32)  # 0..cache_size-1
            angles = values[:, None] * self.freqs[None, :]               # (cache, d/2)

            table = torch.empty(self.cache_size, self.d_model, dtype=torch.float32)
            table[:, 0::2] = torch.sin(angles)
            table[:, 1::2] = torch.cos(angles)
            # 派生バッファなので persistent=False（チェックポイント肥大化を抑える）
            self.register_buffer("sincos_table", table, persistent=False)
        else:
            self.register_buffer("sincos_table", None, persistent=False)

        # 学習可能バイアス（[b_sin^0, b_cos^0, ..., b_sin^{d/2-1}, b_cos^{d/2-1}] に相当） 
        if bias_init == "zeros":
            bias = torch.zeros(self.d_model, dtype=torch.float32)
        elif bias_init == "normal":
            bias = torch.randn(self.d_model, dtype=torch.float32) * float(bias_std)
        else:
            raise ValueError("bias_init must be 'zeros' or 'normal'")

        self.translation_bias = nn.Parameter(bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        inp: 任意形状のテンソル（通常は FMT 値）。torch.long ならテーブル参照を使用可能。
        return: inp.shape + (d_model,) の float32 テンソル
        """
        # cache がある場合: long 入力をそのまま index として使う（高速）
        if (self.sincos_table is not None) and (inp.dtype == torch.long):
            out = self.sincos_table[inp]  # (..., d_model) float32
        else:
            x = inp.to(dtype=torch.float32)
            angles = x.unsqueeze(-1) * self.freqs            # (..., d/2)
            out = torch.empty(*angles.shape[:-1], self.d_model, dtype=torch.float32, device=angles.device)
            out[..., 0::2] = torch.sin(angles)
            out[..., 1::2] = torch.cos(angles)

        # バイアスを加算（broadcast）
        out = out + self.translation_bias.to(dtype=out.dtype)
        return out


class MusicEmbed(nn.Module):
    """
    単一 vocab（REMI 系）に対する埋め込み:
    - pitch / bar / pos は FME に差し替え
    - それ以外は nn.Embedding

    bar_size=1 の場合:
      bar token の出現回数を系列内で cumsum して bar_0, bar_1, ... の絶対 index を作り、FME に入れる。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,

        pitch_start: int = 0,
        pitch_size: int = 128,

        bar_start: int = 128,
        bar_size: int = 1,

        pos_start: int = 129,
        pos_size: int = 32,

        padding_idx: Optional[int] = None,

        # type ごとに base を変える（論文でも属性ごとに異なる B を採用） 
        pitch_base: float = 10000.0,
        bar_base: float = 10000.0,
        pos_base: float = 10000.0,

        # FME bias 初期化
        fme_bias_init: str = "zeros",
        fme_bias_std: float = 0.02,

        # bar の最大個数が事前に分かる場合のみ指定（分からなければ None 推奨）
        bar_cache_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for FME (got {d_model}).")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.pitch_start, self.pitch_size = int(pitch_start), int(pitch_size)
        self.bar_start, self.bar_size = int(bar_start), int(bar_size)
        self.pos_start, self.pos_size = int(pos_start), int(pos_size)
        self.padding_idx = padding_idx

        def _check_range(name: str, start: int, size: int) -> Tuple[int, int]:
            if size <= 0:
                raise ValueError(f"{name}_size must be positive (got {size}).")
            if start < 0:
                raise ValueError(f"{name}_start must be >= 0 (got {start}).")
            end = start + size
            if end > self.vocab_size:
                raise ValueError(f"{name} range [{start}, {end}) exceeds vocab_size={self.vocab_size}.")
            return start, end

        p0, p1 = _check_range("pitch", self.pitch_start, self.pitch_size)
        b0, b1 = _check_range("bar", self.bar_start, self.bar_size)
        s0, s1 = _check_range("pos", self.pos_start, self.pos_size)

        if _ranges_overlap(p0, p1, b0, b1):
            raise ValueError("pitch range overlaps bar range.")
        if _ranges_overlap(p0, p1, s0, s1):
            raise ValueError("pitch range overlaps pos range.")
        if _ranges_overlap(b0, b1, s0, s1):
            raise ValueError("bar range overlaps pos range.")

        self.pitch_end = p1
        self.bar_end = b1
        self.pos_end = s1

        # non-FME（その他 token）用の学習可能埋め込み
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=padding_idx)

        # FME（pitch/pos は値域が小さいので cache を有効化）
        self.pitch_fme = FundamentalMusicEmbedding(
            d_model=self.d_model,
            base=pitch_base,
            bias_init=fme_bias_init,
            bias_std=fme_bias_std,
            cache_size=self.pitch_size,
        )
        self.pos_fme = FundamentalMusicEmbedding(
            d_model=self.d_model,
            base=pos_base,
            bias_init=fme_bias_init,
            bias_std=fme_bias_std,
            cache_size=self.pos_size,
        )
        # bar は最大 bar 数が未知になりがちなのでデフォルトは cache 無し
        self.bar_fme = FundamentalMusicEmbedding(
            d_model=self.d_model,
            base=bar_base,
            bias_init=fme_bias_init,
            bias_std=fme_bias_std,
            cache_size=bar_cache_size,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) LongTensor
        return: (B, T, d_model)
        """
        if idx.dtype != torch.long:
            idx = idx.long()
        if idx.dim() != 2:
            raise ValueError(f"idx must be 2D (B, T). Got shape {tuple(idx.shape)}")

        B, T = idx.shape
        out_dtype = self.token_embed.weight.dtype
        out = torch.empty((B, T, self.d_model), device=idx.device, dtype=out_dtype)

        # pad は常に nn.Embedding 側に回す（pad が FME 範囲に入っても安全）
        pad_mask = idx.eq(self.padding_idx) if self.padding_idx is not None else None

        pitch_mask = (idx >= self.pitch_start) & (idx < self.pitch_end)
        pos_mask = (idx >= self.pos_start) & (idx < self.pos_end)
        bar_mask = (idx >= self.bar_start) & (idx < self.bar_end)

        if pad_mask is not None:
            pitch_mask &= ~pad_mask
            pos_mask &= ~pad_mask
            bar_mask &= ~pad_mask

        fmt_mask = pitch_mask | pos_mask | bar_mask
        other_mask = ~fmt_mask  # pad と「それ以外 token」はここ

        # 1) その他 token（pad 含む）は学習可能埋め込み
        out[other_mask] = self.token_embed(idx[other_mask])

        # 2) pitch: token id -> pitch value (0..pitch_size-1)
        pitch_val = idx[pitch_mask] - self.pitch_start          # long
        out[pitch_mask] = self.pitch_fme(pitch_val).to(dtype=out_dtype)

        # 3) pos: token id -> pos value (0..pos_size-1)
        pos_val = idx[pos_mask] - self.pos_start                # long
        out[pos_mask] = self.pos_fme(pos_val).to(dtype=out_dtype)

        # 4) bar:
        #    bar_size=1 の場合は bar token の出現回数を cumsum して絶対 bar index を作る
        if self.bar_size == 1:
            bar_index = torch.cumsum(bar_mask.to(torch.long), dim=1) - 1  # (B,T) long, bar位置は 0,1,2...
            bar_val = bar_index[bar_mask]                                  # (N_bar,) long
        else:
            bar_val = idx[bar_mask] - self.bar_start

        out[bar_mask] = self.bar_fme(bar_val).to(dtype=out_dtype)

        return out
