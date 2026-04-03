import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import math


def save_checkpoint(
    path: str,
    model: "FrequencyModelV5",
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_acc: float = 0.0,
    extra: Optional[dict] = None,
):
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "temperature": model.temperature.item(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Checkpoint] Saved → {path}  (epoch={epoch}, best_acc={best_acc:.4f})")


def load_checkpoint(
    path: str,
    model: "FrequencyModelV5",
    optimizer=None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    if not os.path.isfile(path):
        print(f"[Checkpoint] No file at {path} — starting from scratch.")
        return {"epoch": 0, "best_acc": 0.0}

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Checkpoint] Missing keys  ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [Checkpoint] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if "temperature" in checkpoint:
        model.set_temperature(checkpoint["temperature"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    last_epoch = checkpoint.get("epoch", -1)
    best_acc   = checkpoint.get("best_acc", 0.0)
    resume_from = last_epoch + 1
    print(f"[Checkpoint] Loaded ← {path}")
    print(f"             Resuming from epoch {resume_from}  "
          f"(best_acc so far = {best_acc:.4f})")
    return {"epoch": resume_from, "best_acc": best_acc}


# ---------------------------------------------------------------------------
# DCT-based frequency decomposer replacing pure FFT
# FFT alone misses mid-frequency compression artifacts that fool the model.
# DCT captures block-DCT patterns (JPEG, DALL-E post-processing) directly.
# ---------------------------------------------------------------------------

class DCTLayer(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        # Build fixed DCT-II basis (no gradients)
        basis = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                basis[k, i] = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        basis[0] *= 1.0 / math.sqrt(n)
        basis[1:] *= math.sqrt(2.0 / n)
        self.register_buffer('basis', basis)  # (n, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — apply 2-D DCT via two 1-D passes
        # Guard against NaN/Inf inputs (can arrive from corrupt batches).
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        # rows (over W dimension)
        x = torch.einsum('ki,bchw->bckw', self.basis, x)
        # cols (over H dimension — was 'bcik' which gave wrong shape)
        x = torch.einsum('kj,bcwj->bcwk', self.basis, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return x


class LearnedFrequencyDecomposer(nn.Module):
    def __init__(self, image_size: int = 128, num_filters: int = 16):
        super().__init__()
        self.image_size = image_size
        self.num_filters = num_filters

        # DCT sub-path — captures JPEG/GAN block artifacts
        self.dct = DCTLayer(image_size)
        self.dct_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
        )

        # FFT sub-path — captures periodic GAN/diffusion grid patterns
        self.freq_conv_real = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
        )
        self.freq_conv_imag = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
        )

        # SRM (Steganalysis Rich Model) residual sub-path
        # High-pass filter that exposes manipulation residuals
        self.srm_conv = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
        )
        self._init_srm_weights()

        # Fuse all three sub-paths: DCT + FFT-real + FFT-imag + SRM = 4 * num_filters
        self.mix = nn.Sequential(
            nn.Conv2d(num_filters * 4, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self._init_weights()

    def _init_srm_weights(self):
        # KV kernel — 5×5 high-pass edge detection core
        srm_kernel = torch.tensor([
            [-1, 2, -2,  2, -1],
            [ 2,-6,  8, -6,  2],
            [-2, 8,-12,  8, -2],
            [ 2,-6,  8, -6,  2],
            [-1, 2, -2,  2, -1],
        ], dtype=torch.float32) / 12.0
        for m in self.srm_conv.modules():
            if isinstance(m, nn.Conv2d):
                with torch.no_grad():
                    for oc in range(m.weight.shape[0]):
                        for ic in range(m.weight.shape[1]):
                            m.weight[oc, ic] = srm_kernel
                break

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)

        # ── FFT branch ────────────────────────────────────────────────────
        # FIX: cast to float32 before FFT to avoid AMP half-precision overflow.
        x_f32 = x.float()

        fft = torch.fft.rfft2(x_f32, norm='ortho')
        real = fft.real
        imag = fft.imag

        # FIX: normalise by per-sample magnitude std to prevent scale blowup.
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        scale = mag.reshape(B, -1).std(dim=1).clamp(min=1e-3).view(B, 1, 1, 1)
        real_n = real / scale
        imag_n = imag / scale

        # Log-magnitude spectrum — preserves energy distribution across freqs.
        # Shift DC to center so low/high frequency regions are spatially meaningful.
        log_mag = torch.log1p(mag)
        lm_mean = log_mag.reshape(B, -1).mean(1).view(B, 1, 1, 1)
        lm_std  = log_mag.reshape(B, -1).std(1).clamp(min=1e-3).view(B, 1, 1, 1)
        log_mag_n = ((log_mag - lm_mean) / lm_std).clamp(-5.0, 5.0)

        # Phase map — captures periodic grid patterns unique to GAN / diffusion upsampling
        phase = torch.atan2(imag_n, real_n + 1e-8)  # already normalised to [-pi, pi]
        phase_n = (phase / 3.14159).clamp(-1.0, 1.0)

        # Pad phase to full spatial size (rfft2 output is H x W//2+1)
        pad_w = self.image_size - log_mag_n.shape[-1]
        if pad_w > 0:
            log_mag_n = F.pad(log_mag_n, (0, pad_w))
            phase_n   = F.pad(phase_n,   (0, pad_w))

        real_spatial = log_mag_n.to(x.dtype)
        imag_spatial = phase_n.to(x.dtype)

        filtered_real = self.freq_conv_real(real_spatial)
        filtered_imag = self.freq_conv_imag(imag_spatial)

        # ── DCT branch ───────────────────────────────────────────────────
        # FIX: compute DCT in float32 and normalise before log to prevent
        # huge coefficient values from blowing up under AMP fp16.
        dct_coeffs = self.dct(x_f32)                            # float32
        # Safe log-magnitude: clamp to avoid log(0) and massive values.
        dct_abs    = torch.abs(dct_coeffs).clamp(min=1e-6, max=1e4)
        dct_log    = torch.log(dct_abs)                          # bounded

        # Per-sample z-score normalisation instead of std division
        # (std can still be very small for near-uniform frequency maps).
        dct_mean = dct_log.reshape(B, -1).mean(dim=1).view(B, 1, 1, 1)
        dct_std  = dct_log.reshape(B, -1).std(dim=1).clamp(min=1e-3).view(B, 1, 1, 1)
        dct_norm = ((dct_log - dct_mean) / dct_std).clamp(-5.0, 5.0).to(x.dtype)

        filtered_dct = self.dct_conv(dct_norm)

        # ── SRM branch ───────────────────────────────────────────────────
        srm_feat = self.srm_conv(x)

        combined = torch.cat([filtered_real, filtered_imag, filtered_dct, srm_feat], dim=1)
        return self.mix(combined)


# ---------------------------------------------------------------------------
# Artifact detector with corrected scale-attention reshaping
# ---------------------------------------------------------------------------

class UpsamplingArtifactDetector(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()

        self.period_small = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.period_mid = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.period_large = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        # Dilated branch: captures global periodic patterns from GAN upsampling
        self.period_dilated = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 48),
            nn.GELU(),
            nn.Linear(48, 192),
            nn.Sigmoid(),
        )

        self.compress = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, freq_feat: torch.Tensor) -> torch.Tensor:
        s  = self.period_small(freq_feat)
        m  = self.period_mid(freq_feat)
        l  = self.period_large(freq_feat)
        d  = self.period_dilated(freq_feat)
        multi = torch.cat([s, m, l, d], dim=1)                # (B, 192, H, W)
        att = self.scale_attention(multi).view(multi.shape[0], 192, 1, 1)
        multi = multi * att
        return self.pool(self.compress(multi))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Backbone with CBAM spatial attention to suppress background noise
# that leads the model to focus on content instead of forensic traces
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx  = self.fc(self.max_pool(x).view(B, C))
        return self.sigmoid((avg + mx).view(B, C, 1, 1))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.act = nn.GELU()

    def forward(self, x):
        out = self.block(x)
        out = out * self.ca(out)
        out = out * self.sa(out)
        return self.act(x + out)


class LightFrequencyBackbone(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(CBAMResBlock(128), CBAMResBlock(128))
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(CBAMResBlock(256), CBAMResBlock(256))
        self.pool = nn.AdaptiveAvgPool2d(1)

        # FIX: project 256-dim backbone output to 256 to match artifact_detector
        # (both streams must be 256 so concat → 512 matches evidence_gate/fusion).
        # Previously the raw pool output was 256 which is correct; no projection
        # needed — but we add an explicit flatten to make the contract obvious.

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        return self.pool(x).flatten(1)   # (B, 256)


# ---------------------------------------------------------------------------
# Main model with evidence gating to reduce hallucination
# ---------------------------------------------------------------------------

class FrequencyModelV5(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 128,
        num_freq_filters: int = 16,
        dropout: float = 0.35,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))

        self.freq_decomposer = LearnedFrequencyDecomposer(
            image_size=image_size,
            num_filters=num_freq_filters,
        )
        self.artifact_detector = UpsamplingArtifactDetector(in_channels=64)  # → (B, 256)
        self.backbone = LightFrequencyBackbone(in_channels=64)               # → (B, 256)

        # Evidence gate: learns to weigh artifact vs backbone streams
        # Prevents one noisy stream from overwhelming the other
        self.evidence_gate = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),   # independent gates — not zero-sum like Softmax
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.classifier = nn.Linear(128, num_classes)
        # Small asymmetric init to break logit symmetry at epoch 0
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)

        # Uncertainty head — high uncertainty triggers abstention at inference
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_head()

    def _init_head(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.uncertainty_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.evidence_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_confidence: bool = False):
        freq_feat     = self.freq_decomposer(x)
        artifact_feat = self.artifact_detector(freq_feat)   # (B, 256)
        backbone_feat = self.backbone(freq_feat)             # (B, 256)

        combined = torch.cat([artifact_feat, backbone_feat], dim=1)  # (B, 512)

        # Evidence gate: soft weighting so neither stream dominates blindly
        gate = self.evidence_gate(combined)                           # (B, 2)

        # FIX: removed the dead/wrong first fused_input assignment.
        # Build fused_input cleanly from gate-weighted streams → (B, 512).
        fused_input = torch.cat([
            artifact_feat * gate[:, 0:1],
            backbone_feat * gate[:, 1:2],
        ], dim=1)   # (B, 512)

        fused = self.fusion(fused_input)
        logits = self.classifier(fused)

        if self.temperature.item() != 1.0:
            #logits = logits / self.temperature
            # Alternative: tensor comparison stays on GPU, no graph break (change 2)
            logits = torch.where(self.temperature != 1.0, logits / self.temperature, logits)

        if return_confidence:
            return logits, self.uncertainty_head(fused)
        return logits

    def set_temperature(self, temperature: float):
        self.temperature.fill_(temperature)

    def calibrate(self, val_loader, device: torch.device):
        import scipy.optimize as opt
        self.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0].to(device), batch[1]
                all_logits.append(self.forward(imgs).cpu())
                all_labels.append(labels)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def nll(T):
            T_val = max(float(T[0]), 1e-2)
            return F.cross_entropy(all_logits / T_val, all_labels).item()

        result = opt.minimize(nll, x0=[1.0], method='Nelder-Mead',
                              options={'xatol': 1e-4, 'maxiter': 200})
        optimal_T = float(max(result.x[0], 1e-2))
        self.set_temperature(optimal_T)
        print(f"Frequency model calibrated: T={optimal_T:.4f}")
        return optimal_T


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_frequency_model_v5(
    num_classes: int = 2,
    image_size: int = 128,
    num_freq_filters: int = 16,
    dropout: float = 0.35,
    temperature: float = 1.0,
    weights_path: Optional[str] = None,
) -> FrequencyModelV5:
    model = FrequencyModelV5(
        num_classes=num_classes,
        image_size=image_size,
        num_freq_filters=num_freq_filters,
        dropout=dropout,
        temperature=temperature,
    )

    if weights_path:
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys ({len(missing)}): "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): "
                      f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            if "temperature" in checkpoint:
                model.set_temperature(checkpoint["temperature"])
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FrequencyModelV5 — Total params: {total:,} | Trainable: {trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Reference training loop
# ---------------------------------------------------------------------------

def train(
    model: FrequencyModelV5,
    train_loader,
    val_loader,
    *,
    device: torch.device,
    total_epochs: int = 100,
    lr: float = 1e-3,
    last_ckpt: str = "checkpoint_last.pth",
    best_ckpt: str = "checkpoint_best.pth",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    info = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
    start_epoch = info["epoch"]
    best_acc    = info["best_acc"]

    if start_epoch >= total_epochs:
        print(f"Training already completed ({start_epoch} / {total_epochs} epochs).")
        return

    model.to(device)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch + 1:>4}/{total_epochs}]  "
              f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  "
              f"best={best_acc:.4f}")

        save_checkpoint(last_ckpt, model, optimizer, scheduler,
                        epoch=epoch, best_acc=best_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_ckpt, model, optimizer, scheduler,
                            epoch=epoch, best_acc=best_acc)
            print(f"  ↑ New best — saved to {best_ckpt}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing FrequencyModelV5...\n")
    model = get_frequency_model_v5()
    model.eval()

    for h, w in [(128, 128), (256, 256), (64, 64)]:
        x = torch.rand(2, 3, h, w)
        with torch.no_grad():
            logits, conf = model(x, return_confidence=True)
        probs = torch.softmax(logits, dim=1)
        print(f"  Input {h}x{w} -> logits {tuple(logits.shape)} | "
              f"probs=[{probs[0,0]:.3f}, {probs[0,1]:.3f}] | "
              f"confidence={conf[0].item():.3f}")

    print("\nGradient check...")
    model.train()
    x = torch.rand(4, 3, 128, 128)
    labels = torch.randint(0, 2, (4,))
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"  Loss={loss.item():.4f} | Backward OK")

    print("\nCheckpoint round-trip test...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "test.pth")
        opt  = torch.optim.AdamW(model.parameters())
        save_checkpoint(ckpt, model, opt, epoch=7, best_acc=0.923)
        info = load_checkpoint(ckpt, model, opt)
        assert info["epoch"] == 8,    f"Expected 8, got {info['epoch']}"
        assert abs(info["best_acc"] - 0.923) < 1e-6
        print(f"  Saved epoch=7  →  resumed from epoch={info['epoch']}  ✓")

    print("\nAll tests passed!")