Guo ら（AAAI 2023）の **Fundamental Music Embedding（FME）** を適用したい。
FME は、Transformer の標準的なサイン・コサイン符号化（Sinusoidal Encoding）を “背骨” にしつつ、**トークン種別ごとに学習可能なバイアス（bias）** を付与して、pitch / onset など異種属性の埋め込み空間を分離可能にする。
入力は単一 token id（語彙 1 本）REMI系で、pitch/bar/pos が token 範囲で区別される。

    vocab_size : int = 164
    pitch_start : int = 0
    pitch_size : int = 128
    bar_start : int = 128
    bar_size : int = 1
    pos_start : int = 129
    pos_size : int = 32

`idx`（B,T）から該当 token をマスクして、該当部分だけ FME を適用し，それ以外はnn.embedを適用する。
pitch bar posにFMEを適用する。
ただしbar sizeは1だがカウントしてbar_0 bar_1 bar_2とabsolute embedする

class FundamentalMusicEmbedding(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        self.d_model = d_model
        # 1. 計算済みの角度係数をバッファとして登録
        # これにより model.to(device) した際に自動的にGPUへ移動し、state_dictにも保存される
        i = torch.arange(d_model)
        exponent = (2 * (i // 2)) / d_model
        angle_rates = 1 / torch.pow(base, exponent)
        self.register_buffer('angle_rates', angle_rates.view(1, 1, -1))
        # 2. 学習可能なバイアス
        self.translation_bias = nn.Parameter(torch.randn(1, 1, d_model))
    def forward(self, inp):
        # inp shape: (batch, num_pitch)
        # ブロードキャストを利用して角度を計算 (batch, num_pitch, d_model)
        x = inp.unsqueeze(-1)
        angle_rads = x * self.angle_rates
        # sin/cos の適用
        # メモリ効率のため empty_like を使用し、in-placeで代入
        pos_encoding = torch.empty_like(angle_rads)
        pos_encoding[..., 0::2] = torch.sin(angle_rads[..., 0::2])
        pos_encoding[..., 1::2] = torch.cos(angle_rads[..., 1::2])
        # バイアスの加算
        out = pos_encoding + self.translation_bias

        nn.embedのかわりとなる　musicembed classを作成

            out = self.linear(out)
            
        return out
