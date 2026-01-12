# GPT-MIDI Training Hyperparameter Plan (909曲 / 平均2000トークン / A100 40GB / tpq=8)

## 前提
- 909曲（train+val合計）
- 平均トークン数 ≈ 2000
- GPU: A100 40GB
- tpq=8
- EOF_ID=163 をドキュメント境界として使用

---

## 調整方針（要点）
- **sequence_length** を 2048〜4096 に縮小
- **学習率** を全体的に下げる（小規模データ向け）
- **val_tokens** を縮小して評価コストを抑制
- **モデル規模** を縮小して過学習リスクを低減
- **attention blocksize** を短い系列向けにクランプ
- **weight_decay** を軽く導入

---

## 具体調整候補

### 1) sequence_length と batch_size
- 目安: **T=2048**（必要なら 4096 へ拡大）
- global batch を維持するため、**batch_size と grad accumulation を再調整**
- `val_tokens % (T * ddp_world_size) == 0` を満たすように設定

### 2) val_tokens / val_loss_every
- 平均2000トークン × 909曲 ≈ 180万トークン規模
- `val_tokens`: **200k〜1M** に縮小
- `val_loss_every`: **200〜500** 程度に引き上げ

### 3) 学習率
- `optimizer1` (wte/vte): **0.1〜0.2**
- `optimizer2` (lm_head): **0.002〜0.004**
- `optimizer3` (Muon): **0.01〜0.02**
- `warmup_iters`: **200〜500** へ増やす

### 4) モデル規模
- `n_layer`: **6〜8**
- `n_embd`: **256〜384**
- `n_head`: **4〜6**
- `vte` 埋め込みが大きすぎる場合は **n_embd * 4〜8** に調整検討

### 5) attention blocksize
- 短い系列向けに **`min(T, 1024)` などでクランプ**
- `T` が小さい場合は固定値（例: 512 or 1024）も可

### 6) weight_decay
- `weight_decay`: **0.01** 程度を導入
- 必要なら wte/vte のみ 0 にする案も検討

---

## 実装タスク（概要）
- `sequence_length` / `batch_size` / `val_tokens` を再設定
- 学習率をデータ規模に合わせて下げる
- モデル規模（層数・埋め込み）を縮小
- attention blocksize を短い系列向けに調整
- weight_decay の導入

---

## 参考値（初期案）
- **T=2048**
- **batch_size=32**（GPU枚数に応じて調整）
- **val_tokens=500k**
- **n_layer=6 / n_embd=256 / n_head=4**
- **LR: wte/vte=0.15, lm_head=0.003, Muon=0.015**

