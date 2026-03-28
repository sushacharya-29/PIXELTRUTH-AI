"""
Frequency Domain CNN Model for AI-Generated Image Detection
============================================================

v3.1 — Fixes two crash bugs and one silent correctness bug.

Root causes fixed over v3.0:
  1. FFTPreprocessor crashes under torch.amp.autocast — CRASH BUG:
     torch.fft.fft2 and torch.quantile do NOT support float16 (Half) tensors.
     Under mixed-precision training, autocast casts the model's input to
     float16 before entering forward(). The FFTPreprocessor received this
     Half tensor and immediately crashed with:
       RuntimeError: expected scalar type Float but got Half
     Fix: cast x to float32 at the start of FFTPreprocessor.forward(),
     compute all FFT and quantile operations in float32, then cast the
     normalised magnitude back to the original input dtype (so the rest
     of the network continues in whatever dtype autocast chose).

  2. temperature not persisted in state_dict — SILENT CORRECTNESS BUG:
     Same bug as spatial model (v3.1). self.temperature was a plain Python
     float — not saved in state_dict. After load_state_dict(), temperature
     silently reverted to 1.0, causing miscalibrated inference.
     Fix: self.register_buffer('temperature', torch.tensor(temperature))

v3.0 fixes (inherited):
  1. Classifier output bias initialized toward REAL.
  2. Single dropout layer (p=0.25).
  3. quantile() reshape simplified for PyTorch 2.x compat.
  4. Temperature scaling support added.

Architecture: unchanged from v3.0.
    Input  : (B, 3, 128, 128)  — raw pixel tensor in [0, 1], NOT ImageNet-normalised
    FFT    : per-channel 2D FFT → log-magnitude → percentile-clamp normalise
    CNN    : 4 stages (32→64→128→256) with residual blocks + single CBAM at bottleneck
    Output : (B, 2)            — logits [FAKE, REAL]
              class 0 = FAKE (AI-Generated)  ← alphabetical CIFAKE order
              class 1 = REAL

CRITICAL: This model expects raw [0,1] pixel tensors as input.
          Do NOT apply ImageNet/CIFAKE normalisation before this model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Frequency model must receive raw [0,1] tensors (ToTensor only, NO Normalize).
FREQ_MODEL_MEAN = None  # intentionally None — do not normalise
FREQ_MODEL_STD  = None


class FFTPreprocessor(nn.Module):
    """
    Vectorised 2D FFT magnitude extractor.

    v3.0 fix: simplified quantile reshape logic for cross-version compatibility.
    """

    def __init__(self, log_scale: bool = True, clip_percentile: float = 99.5):
        super().__init__()
        self.log_scale       = log_scale
        self.clip_percentile = clip_percentile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — raw pixel values in [0, 1]

        Returns:
            magnitude: (B, 3, H, W)  — normalised log-magnitude spectra in [0, 1]
        """
        # BUG FIX: torch.fft.fft2 and torch.quantile do NOT support float16.
        # Under torch.amp.autocast the input tensor arrives as Half (float16).
        # We must compute FFT in float32 and cast the result back to the
        # original dtype so the rest of the model stays in the autocast dtype.
        input_dtype = x.dtype
        x = x.float()                                   # ensure float32 for FFT

        fft = torch.fft.fft2(x, norm='ortho')          # (B, 3, H, W) complex
        fft = torch.fft.fftshift(fft, dim=(-2, -1))    # shift DC to centre
        mag = torch.abs(fft)                            # (B, 3, H, W) real

        if self.log_scale:
            mag = torch.log1p(mag)

        B, C, H, W = mag.shape
        flat = mag.view(B * C, H * W)                  # (B*C, H*W) — simpler

        # FIX 3: Simplified quantile call — works across all PyTorch versions
        q    = torch.quantile(flat, self.clip_percentile / 100.0, dim=1)  # (B*C,)
        hi   = q.view(B, C, 1, 1)                      # (B, C, 1, 1)
        mag  = torch.clamp(mag, max=hi.expand_as(mag))

        # Min-max normalise per (B, C)
        flat = mag.view(B, C, -1)
        lo   = flat.min(dim=2).values.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        hi2  = flat.max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
        denom = (hi2 - lo).clamp(min=1e-6)
        mag  = (mag - lo) / denom

        return mag.to(input_dtype)  # (B, 3, H, W) in [0, 1], restored to input dtype


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FreqResBlock(nn.Module):
    """Residual block for frequency-domain feature processing."""
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


class CBAM(nn.Module):
    """CBAM attention — identical to spatial model for consistency."""
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.max_pool    = nn.AdaptiveMaxPool2d(1)
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
        x   = x * self.sigmoid(avg + mx)
        avg_s = x.mean(1, keepdim=True)
        max_s, _ = x.max(1, keepdim=True)
        x   = x * self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], 1)))
        return x


class FrequencyDomainCNN(nn.Module):
    """
    Frequency Domain CNN for AI Image Detection — v3.0

    Key v3.0 fix: bias initialization + single dropout layer.

    CRITICAL: This model expects raw [0,1] pixel tensors as input.
              Do NOT apply ImageNet normalisation before this model.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.25,
                 process_in_model: bool = True, temperature: float = 1.0):
        super().__init__()
        self.process_in_model = process_in_model
        # BUG FIX: register temperature as a buffer so it is included in
        # state_dict and correctly restored on load_state_dict().
        # Previously it was a plain Python float — silently lost on save/load.
        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))

        if process_in_model:
            self.fft_preprocessor = FFTPreprocessor(log_scale=True, clip_percentile=99.5)

        self.stem = nn.Sequential(
            ConvBnRelu(3, 32, kernel=3, stride=1, padding=1),
            ConvBnRelu(32, 32, kernel=3, stride=1, padding=1),
        )

        self.stage1 = nn.Sequential(
            ConvBnRelu(32, 64, stride=2),
            FreqResBlock(64),
        )

        self.stage2 = nn.Sequential(
            ConvBnRelu(64, 128, stride=2),
            FreqResBlock(128),
            FreqResBlock(128),
        )

        self.attention = CBAM(128, reduction=8)

        self.stage3 = nn.Sequential(
            ConvBnRelu(128, 256, stride=2),
            FreqResBlock(256),
            FreqResBlock(256),
        )

        self.stage4 = nn.Sequential(
            ConvBnRelu(256, 256, stride=2),
            FreqResBlock(256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # FIX 2: Single dropout — eliminates train/inference compounding gap
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
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

        # FIX 1: Initialize final output bias to counteract FAKE-only collapse
        final_linear = None
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is not None and final_linear.bias is not None:
            with torch.no_grad():
                final_linear.bias[0] = -0.2   # FAKE: slight negative push
                final_linear.bias[1] =  0.2   # REAL: slight positive push

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128)
               — raw [0,1] pixel values if process_in_model=True
               — pre-computed FFT magnitude if process_in_model=False

        Returns:
            logits: (B, 2)  [FAKE_logit, REAL_logit]
        """
        if self.process_in_model:
            x = self.fft_preprocessor(x)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.attention(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if self.temperature.item() != 1.0:
            logits = logits / self.temperature

        return logits

    def set_temperature(self, temperature: float):
        """Post-hoc calibration — increase above 1.0 if overconfident on FAKE."""
        # Write into the existing buffer tensor (keeps it on the correct device)
        self.temperature.fill_(temperature)

    def extract_frequency_features(self, x: torch.Tensor) -> dict:
        """Extract feature maps at each stage for visualisation / GradCAM."""
        feats = {}
        if self.process_in_model:
            x = self.fft_preprocessor(x)
            feats['fft_input'] = x
        x = self.stem(x);      feats['stem']    = x
        x = self.stage1(x);    feats['stage1']  = x
        x = self.stage2(x);    feats['stage2']  = x
        x = self.attention(x); feats['cbam']    = x
        x = self.stage3(x);    feats['stage3']  = x
        x = self.stage4(x);    feats['stage4']  = x
        return feats


def get_frequency_model(
    num_classes: int = 2,
    dropout: float = 0.25,
    pretrained: bool = False,
    weights_path: str = None,
    process_in_model: bool = True,
    temperature: float = 1.0,
) -> FrequencyDomainCNN:
    """
    Factory function for the Frequency Domain CNN.

    ── CRITICAL: DataLoader transform for this model ──────────────────────
    Because FFT is performed INSIDE the model, your training DataLoader must
    NOT apply ImageNet normalisation. Use this transform instead:

        import torchvision.transforms as T
        freq_transform_train = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),          # → [0,1], STOP HERE
        ])
        freq_transform_val = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
    ───────────────────────────────────────────────────────────────────────

    Training recommendations:
      - loss: CrossEntropyLoss(label_smoothing=0.1, weight=[1.0, 2.0])
        The 2.0 weight on REAL is critical to prevent FAKE-only collapse.
      - optimizer: AdamW, lr=3e-4, weight_decay=1e-4
      - gradient clipping: clip_grad_norm_(model.parameters(), 1.0)
      - scheduler: CosineAnnealingLR(T_max=num_epochs)
      - best model: balanced_accuracy (NOT overall accuracy)
    """
    model = FrequencyDomainCNN(
        num_classes=num_classes,
        dropout=dropout,
        process_in_model=process_in_model,
        temperature=temperature,
    )

    if pretrained and weights_path:
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state, strict=True)
            print(f"✓ Loaded frequency weights from {weights_path}")
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
            print(f"⚠ Could not load frequency weights: {exc}")

    return model


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = get_frequency_model(process_in_model=True)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}  (~{total*4/1e6:.1f} MB fp32)")

    # Check bias initialization
    final_linear = None
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            final_linear = m
    print(f"Classifier output bias: {final_linear.bias.data.tolist()}")
    print(f"  (expected: FAKE≈-0.2, REAL≈+0.2)")

    # Simulate raw [0,1] pixel input (NOT normalised)
    x = torch.rand(4, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"Probs  : FAKE={probs[0,0]:.3f}  REAL={probs[0,1]:.3f}")

    # FFT preprocessor standalone test
    preprocessor = FFTPreprocessor()
    with torch.no_grad():
        fft_out = preprocessor(x)
    print(f"FFT output range: [{fft_out.min():.4f}, {fft_out.max():.4f}]  (expected [0,1])")
    print("✓ Frequency model v3.0 self-test passed.")