import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from config import cfg
except ImportError:
    from .config import cfg


class VibrationEncoder(nn.Module):
    """
    Encoder for vibration wavelet features.
    Input : [batch, seq_len, vib_feature_dim]
    Output: [batch, seq_len, embed_dim]
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        channels=(64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for out_channels in channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.cnn(x)        # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.proj(x)
        x = self.norm(x)
        return x


class ThermalEncoder(nn.Module):
    """
    Encoder for temperature features.
    Input : [batch, seq_len, temp_feature_dim]
    Output: [batch, seq_len, embed_dim]
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module.
    Vibration embedding serves as Query, Thermal embedding serves as Key/Value.
    Recalibrates vibration representation using thermal degradation context.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, vib, temp):
        attn_out, attn_weights = self.cross_attn(
            query=vib,
            key=temp,
            value=temp,
            need_weights=True,
        )

        x = self.norm1(vib + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        fused = self.norm2(x + self.dropout(ffn_out))

        return fused, attn_weights


class TemporalModel(nn.Module):
    """
    Temporal degradation modeling using GRU / LSTM / BiLSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.output_dim = hidden_dim

        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.output_dim = hidden_dim

        elif self.rnn_type == "bilstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.output_dim = hidden_dim * 2

        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Use 'gru', 'lstm', or 'bilstm'.")

    def forward(self, x):
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        return h_last


class RegressionHead(nn.Module):
    """
    MLP Regression Head for RUL prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WCAGRUModel(nn.Module):
    """
    WCA-GRU Model Architecture for Bearing RUL Prediction.

    Supports ablation modes via cfg.MODALITY_MODE:
        - "wca_gru"   : Proposed Wavelet Cross-Attention GRU
        - "vib_only"  : Vibration-only baseline (Case 1)
        - "temp_only" : Temperature-only baseline (Case 2)
        - "concat"    : Concatenation baseline (Case 3)
    """

    def __init__(self, config=cfg):
        super().__init__()

        self.config = config
        self.modality_mode = config.MODALITY_MODE.lower()
        if self.modality_mode not in {"wca_gru", "vib_only", "temp_only", "concat"}:
            raise ValueError(f"Unsupported modality mode: {self.modality_mode}")

        self.vib_encoder = VibrationEncoder(
            input_dim=config.VIB_FEATURE_DIM,
            embed_dim=config.FUSION_DIM,
            channels=config.CNN_CHANNELS,
            kernel_size=config.CNN_KERNEL_SIZE,
            dropout=config.CNN_DROPOUT,
        )

        self.temp_encoder = ThermalEncoder(
            input_dim=config.TEMP_FEATURE_DIM,
            embed_dim=config.FUSION_DIM,
            hidden_dim=config.TEMP_EMBED_DIM,
            dropout=config.CNN_DROPOUT,
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=config.FUSION_DIM,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.ATTENTION_DROPOUT,
        )

        # Single-modality baselines do not retain an inactive encoder or fusion.
        if self.modality_mode == "vib_only":
            self.temp_encoder = None
        elif self.modality_mode == "temp_only":
            self.vib_encoder = None
        if self.modality_mode != "wca_gru":
            self.fusion = None

        self.temporal = TemporalModel(
            input_dim=config.FUSION_DIM * (2 if self.modality_mode == "concat" else 1),
            hidden_dim=config.RNN_HIDDEN_DIM,
            num_layers=config.RNN_LAYERS,
            dropout=config.RNN_DROPOUT,
            rnn_type=config.RNN_TYPE,
        )

        self.regressor = RegressionHead(
            input_dim=self.temporal.output_dim,
            hidden_dim=config.FC_HIDDEN_DIM,
            dropout=config.FC_DROPOUT,
        )

    def forward(self, vib_x, temp_x, return_attention=False):
        attn_weights = None
        if self.modality_mode == "vib_only":
            fused_z = self.vib_encoder(vib_x)
        elif self.modality_mode == "temp_only":
            fused_z = self.temp_encoder(temp_x)
        else:
            vib_z = self.vib_encoder(vib_x)
            temp_z = self.temp_encoder(temp_x)
            if self.modality_mode == "concat":
                fused_z = torch.cat((vib_z, temp_z), dim=-1)
            else:
                fused_z, attn_weights = self.fusion(vib_z, temp_z)

        temporal_z = self.temporal(fused_z)
        pred_rul = self.regressor(temporal_z)

        if return_attention:
            return pred_rul, attn_weights

        return pred_rul


def build_model(config=cfg):
    return WCAGRUModel(config)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    cfg.make_dirs()
    model = build_model(cfg)
    total_params, trainable_params = count_parameters(model)

    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 4
    dummy_vib = torch.randn(batch_size, cfg.SEQ_LEN, cfg.VIB_FEATURE_DIM)
    dummy_temp = torch.randn(batch_size, cfg.SEQ_LEN, cfg.TEMP_FEATURE_DIM)

    with torch.no_grad():
        y = model(dummy_vib, dummy_temp)

    print("\nDummy output shape:", y.shape)
