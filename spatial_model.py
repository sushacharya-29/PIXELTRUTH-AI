"""
Spatial Domain CNN Model for AI-Generated Image Detection
==========================================================

v3.1 — Fixes temperature persistence bug.

Root causes fixed over v3.0:
  1. temperature not persisted in state_dict (SILENT CORRECTNESS BUG):
     self.temperature was stored as a plain Python float attribute.
     Plain float attributes are NOT included in state_dict, so any
     temperature value set via set_temperature() or passed to the
     constructor was silently lost on save/load. After loading a
     checkpoint, temperature would always revert to 1.0 regardless
     of what was saved.
     Fix: use self.register_buffer('temperature', torch.tensor(temperature))
     so the value travels with the checkpoint automatically.

v3.0 fixes (inherited):
  1. Classifier output bias initialized to counteract FAKE-only collapse.
  2. Single dropout layer (p=0.3) replaces compounded two-layer dropout.
  3. Temperature scaling support added to forward().

Architecture: unchanged.
    Input:  (batch, 3, 128, 128)
    Output: (batch, 2)  — logits [FAKE, REAL]
    class 0 = FAKE (AI)  ← CIFAKE ImageFolder alphabetical order
    class 1 = REAL

CIFAKE normalization (NOT ImageNet):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# CIFAKE-CORRECT normalization (NOT ImageNet)
# ──────────────────────────────────────────────────────────────
CIFAKE_MEAN = [0.4914, 0.4822, 0.4465]
CIFAKE_STD  = [0.2470, 0.2435, 0.2616]


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Applied once at the bottleneck — not on every block.
    """
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        pad = (spatial_kernel - 1) // 2
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.channel_mlp(self.avg_pool(x))
        mx  = self.channel_mlp(self.max_pool(x))
        ca  = self.sigmoid(avg + mx)
        x   = x * ca
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa  = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class ConvBnRelu(nn.Module):
    """Standard Conv→BN→ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block without per-block attention."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class SpatialDomainCNN(nn.Module):
    """
    Spatial Domain CNN for AI Image Detection — v3.0

    Channel progression: 3 → 32 → 64 → 128 → 192 → 256
    Downsampling:  128 → 64 → 32 → 16 → 8
    CBAM attention at 192-channel bottleneck.

    Key fix: classifier bias initialized to counteract FAKE-only collapse.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 temperature: float = 1.0):
        super().__init__()
        # BUG FIX: register temperature as a buffer so it is included in
        # state_dict and correctly restored on load_state_dict().
        # Previously it was a plain Python float — silently lost on save/load,
        # causing miscalibrated inference after loading a tuned checkpoint.
        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))

        # ── Stem ──
        self.stem = nn.Sequential(
            ConvBnRelu(3,  32, kernel=3, stride=1, padding=1),
            ConvBnRelu(32, 32, kernel=3, stride=1, padding=1),
        )

        # ── Stage 1: 128 → 64 ──
        self.stage1 = nn.Sequential(
            ConvBnRelu(32, 64, stride=2),
            ResBlock(64),
        )

        # ── Stage 2: 64 → 32 ──
        self.stage2 = nn.Sequential(
            ConvBnRelu(64, 128, stride=2),
            ResBlock(128),
            ResBlock(128),
        )

        # ── Stage 3: 32 → 16 ──
        self.stage3 = nn.Sequential(
            ConvBnRelu(128, 192, stride=2),
            ResBlock(192),
            ResBlock(192),
        )

        # ── CBAM at bottleneck ──
        self.attention = CBAM(192, reduction=8)

        # ── Stage 4: 16 → 8 ──
        self.stage4 = nn.Sequential(
            ConvBnRelu(192, 256, stride=2),
            ResBlock(256),
        )

        # ── Classifier ──
        # FIX 2: Single dropout instead of compounded two-layer dropout
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),          # single dropout — removes train/inference gap
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),  # bias initialized in _initialize_weights
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # FIX 1: Bias the final classifier output toward REAL to counteract
        # the natural tendency to over-predict FAKE.
        # FAKE output (index 0) gets a slight negative nudge.
        # REAL output (index 1) gets a slight positive nudge.
        # This is intentionally small — the model should learn the right bias
        # during training, but this initialization prevents early collapse.
        final_linear = None
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is not None and final_linear.bias is not None:
            with torch.no_grad():
                final_linear.bias[0] = -0.2   # FAKE logit: slight negative push
                final_linear.bias[1] =  0.2   # REAL logit: slight positive push

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) — normalized with CIFAKE_MEAN / CIFAKE_STD

        Returns:
            logits: (B, 2)  — [FAKE_logit, REAL_logit]
                    class 0 = FAKE (AI-Generated)
                    class 1 = REAL
        """
        x = self.stem(x)        # (B,  32, 128, 128)
        x = self.stage1(x)      # (B,  64,  64,  64)
        x = self.stage2(x)      # (B, 128,  32,  32)
        x = self.stage3(x)      # (B, 192,  16,  16)
        x = self.attention(x)   # (B, 192,  16,  16)
        x = self.stage4(x)      # (B, 256,   8,   8)
        x = self.global_pool(x) # (B, 256,   1,   1)
        x = torch.flatten(x, 1) # (B, 256)
        logits = self.classifier(x)

        # Temperature scaling: values > 1.0 soften the distribution (reduces overconfidence)
        # Note: temperature is a registered buffer (tensor), so compare with .item()
        if self.temperature.item() != 1.0:
            logits = logits / self.temperature

        return logits

    def set_temperature(self, temperature: float):
        """
        Set temperature for post-hoc calibration.

        If the model is over-confident on FAKE (assigns >95% to FAKE for real images),
        increase temperature above 1.0 (e.g., 1.5-2.0) to soften predictions.

        To find the optimal temperature, minimize NLL on a calibration set:
            from torch.optim import LBFGS
            temp_model = SpatialDomainCNN(...)
            # optimize temp_model.temperature on val set
        """
        # Write into the existing buffer tensor (keeps it on the correct device)
        self.temperature.fill_(temperature)

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate feature maps for debugging / GradCAM."""
        feats = {}
        x = self.stem(x);      feats['stem']    = x
        x = self.stage1(x);    feats['stage1']  = x
        x = self.stage2(x);    feats['stage2']  = x
        x = self.stage3(x);    feats['stage3']  = x
        x = self.attention(x); feats['cbam']    = x
        x = self.stage4(x);    feats['stage4']  = x
        return feats


def get_spatial_model(
    num_classes: int = 2,
    dropout: float = 0.3,
    pretrained: bool = False,
    weights_path: str = None,
    temperature: float = 1.0,
) -> SpatialDomainCNN:
    """
    Factory function for the Spatial Domain CNN.

    IMPORTANT — CIFAKE label ordering (ImageFolder sorts alphabetically):
        index 0 → 'FAKE'  (AI-Generated images)
        index 1 → 'REAL'  (Real photographs)

    Training recommendations:
      - optimizer: AdamW, lr=3e-4, weight_decay=1e-4
      - loss: CrossEntropyLoss(label_smoothing=0.1, weight=[1.0, 2.0])
        The 2.0 weight on REAL is critical to prevent FAKE-only collapse.
      - gradient clipping: clip_grad_norm_(model.parameters(), max_norm=1.0)
      - scheduler: CosineAnnealingLR(T_max=num_epochs)
      - best model: save on balanced_accuracy, not overall accuracy
      - augmentation: HorizontalFlip + mild brightness/contrast ONLY
        (NO blur, NO rotation — destroy pixel-level artifacts)

    Temperature calibration (post-training):
      If model is over-confident on FAKE: set temperature=1.5 or 2.0.
      This does NOT require retraining — just set it before inference.
    """
    model = SpatialDomainCNN(num_classes=num_classes, dropout=dropout,
                             temperature=temperature)

    if pretrained and weights_path:
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state, strict=True)
            print(f"✓ Loaded spatial weights from {weights_path}")
            if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
                m = checkpoint['metrics']
                fake_acc = m.get('fake_accuracy', 'N/A')
                real_acc = m.get('real_accuracy', 'N/A')
                bal_acc  = m.get('balanced_accuracy', 'N/A')
                print(f"  Checkpoint metrics — FAKE={fake_acc}%  REAL={real_acc}%  balanced={bal_acc}%")
                if isinstance(real_acc, float) and real_acc < 70:
                    print(f"  ⚠ REAL accuracy in checkpoint is low ({real_acc}%).")
                    print(f"    Consider retraining with higher real_weight or setting temperature > 1.0")
        except Exception as exc:
            print(f"⚠ Could not load spatial weights: {exc}")

    return model


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = get_spatial_model()
    total  = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}  (~{total*4/1e6:.1f} MB fp32)")

    # Check bias initialization
    final_linear = None
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            final_linear = m
    print(f"Classifier output bias: {final_linear.bias.data.tolist()}")
    print(f"  (expected: FAKE≈-0.2, REAL≈+0.2)")

    x = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"Probs  : FAKE={probs[0,0]:.3f}  REAL={probs[0,1]:.3f}")
    print("✓ Spatial model v3.0 self-test passed.")