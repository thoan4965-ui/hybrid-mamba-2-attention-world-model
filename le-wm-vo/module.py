import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
# Dùng thư viện ncps chính thức (tác giả gốc CfC: Mathias Lechner)
# pip install ncps
from ncps.torch import CfC
from ncps.wirings import FullyConnected

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
        use_norm=False,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(smoothed_dim) if use_norm else nn.Identity()
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class CfCPredictor(nn.Module):
    """
    CfC-based predictor dùng thư viện ncps chính thức.

    Wiring: FullyConnected (tương đương dense CfC cũ, không có sparse NCP).
    use_mixed=False: phù hợp cho dataset nhỏ (5000 samples), tránh overfit.
    timespans=1.0: đồng đều vì data thu ở 15fps cố định.
    """

    def __init__(
        self,
        *,
        num_frames,
        input_dim,          # embedding dim của pixels (D)
        hidden_dim,         # số neurons của CfC (units)
        output_dim=None,    # đầu ra (mặc định là input_dim)
        action_dim=192,     # embedding dim của action
        use_mixed=False,    # False: chỉ CfC. True: LSTM+CfC (không cần với data nhỏ)
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim or input_dim

        # CfC với batch_first=True để khớp (B, T, D)
        # units=hidden_dim và proj_size=self.output_dim để project ra output_dim
        # backbone_layers=2 và backbone_units=256 để khớp với weights_epoch_34.pt checkpoint
        self.cfc = CfC(
            input_size=input_dim + action_dim,
            units=hidden_dim,
            proj_size=self.output_dim,
            backbone_layers=2,
            backbone_units=256,
            batch_first=True,
            mixed_memory=use_mixed,
        )

    def forward(self, x, c):
        """
        x: (B, T, D) - pixel embeddings
        c: (B, T, A_emb) - action embeddings
        """
        # Nối pixel embedding và action embedding
        inputs = torch.cat([x, c], dim=-1)  # (B, T, D + A_emb)

        # timespans=None → ncps tự đặt ts=1.0 (chuẩn cho 15fps đều)
        # output: (B, T, output_dim) | h_n: hidden state cuối
        out, _ = self.cfc(inputs, timespans=None)
        return out  # (B, T, output_dim)


class CfCPredictorV2(nn.Module):
    """
    CfC predictor — class duy nhất cho mọi CfC variant (V1/V2/V3).
    Config variants (params qua constructor, không subclass riêng):

    | Variant    | hidden_dim | backbone_layers | backbone_units | Total params |
    |------------|-----------|-----------------|----------------|-------------|
    | CfC V1     | 64        | 1               | 64             | 27K         |
    | CfC V2/V3  | 96        | 1               | 96             | 55.8K (≈AR) |
    | (sai)      | 128       | 2               | 128            | 111K (2×AR) |

    Dataset action format: single-frame 8-DOF (GPUDataset, KHÔNG stack frameskip).
    action_encoder.input_dim=8 cho CfC (khác AR dùng 24 stacked).
    """

    def __init__(
        self,
        *,
        num_frames,
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        action_dim=32,
        backbone_layers=1,
        backbone_units=64,
        use_mixed=False,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.cfc = CfC(
            input_size=input_dim + action_dim,
            units=hidden_dim,
            proj_size=output_dim,
            backbone_layers=backbone_layers,
            backbone_units=backbone_units,
            batch_first=True,
            mixed_memory=use_mixed,
        )

    def forward(self, x, c):
        """
        x: (B, T, D) - pixel embeddings
        c: (B, T, A_emb) - action embeddings
        """
        inputs = torch.cat([x, c], dim=-1)
        out, _ = self.cfc(inputs, timespans=None)
        return out

    def step(self, x, c, h=None, timespan=None):
        """
        Single step forward with hidden state carry.

        Args:
            x: (B, 1, D) - pixel embedding at current step
            c: (B, 1, A_emb) - action embedding at current step
            h: hidden state from previous step (None for first step)
            timespan: (B, 1, 1) or scalar for ODE time constant

        Returns:
            out: (B, 1, output_dim) - predicted next embedding
            h: updated hidden state for next step
        """
        inputs = torch.cat([x, c], dim=-1)
        ts = None if timespan is None else timespan
        out, h = self.cfc(inputs, timespans=ts, hx=h)
        return out, h


class TinyViT(nn.Module):
    """
    Custom tiny ViT cho task grasp 8 DOF.
    Train từ scratch, không pretrained.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        mlp_dim: int = 256,
        output_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Patch embedding: (B, 3, H, W) -> (B, hidden_dim, H', W')
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Position embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(hidden_dim, heads=num_heads, dim_head=16, dropout=dropout),
                FeedForward(hidden_dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, 3, H, W) - RGB image
        Returns: (B, output_dim) - latent embedding
        """
        B = x.shape[0]
        
        # Patch embedding: (B, 3, H, W) -> (B, hidden_dim, H', W')
        x = self.patch_embed(x)
        
        # Flatten: (B, hidden_dim, H', W') -> (B, hidden_dim, N) -> (B, N, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token: (B, N, hidden_dim) -> (B, N+1, hidden_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for attn, mlp in self.blocks:
            x = x + attn(x, causal=False)
            x = x + mlp(x)
        
        # Final norm
        x = self.norm(x)
        
        # Extract CLS token: (B, N+1, hidden_dim) -> (B, hidden_dim)
        x = x[:, 0]
        
        # Output projection: (B, hidden_dim) -> (B, output_dim)
        x = self.output_proj(x)
        
        return x


class ResNet10tEncoder(nn.Module):
    """
    ResNet10t-based visual encoder using timm library.
    It takes an image tensor of shape (B, 3, 224, 224) and outputs (B, 512).
    """

    def __init__(
        self,
        *,
        pretrained=True,
        freeze_early_layers=True,
        **kwargs,
    ):
        super().__init__()
        import timm
        # num_classes=0 removes the final linear classification layer and returns pooled features (512 dimensions)
        self.backbone = timm.create_model("resnet10t", pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features # 512
        
        # Freezing early layers (all except layer4)
        if freeze_early_layers:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4"):
                    param.requires_grad = False

    def forward(self, x, **kwargs):
        """
        x: (B, 3, 224, 224)
        """
        # timm resnet10t outputs (B, 512) directly when num_classes=0
        return self.backbone(x)

