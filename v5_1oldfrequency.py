import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: "FrequencyModelV5",
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_acc: float = 0.0,
    extra: Optional[dict] = None,
):
    """Save a full training checkpoint (model + optimiser + epoch counter)."""
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
    """
    Load a checkpoint and restore model (+ optionally optimiser/scheduler) state.

    Returns a dict with at least:
        - 'epoch'    : int   — the epoch to *resume from* (last completed + 1)
        - 'best_acc' : float — best validation accuracy seen so far
    """
    if not os.path.isfile(path):
        print(f"[Checkpoint] No file at {path} — starting from scratch.")
        return {"epoch": 0, "best_acc": 0.0}

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # ── model weights ──────────────────────────────────────────────────────
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Checkpoint] Missing keys  ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [Checkpoint] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # ── temperature ────────────────────────────────────────────────────────
    if "temperature" in checkpoint:
        model.set_temperature(checkpoint["temperature"])

    # ── optimiser ─────────────────────────────────────────────────────────
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ── scheduler ─────────────────────────────────────────────────────────
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ── epoch bookkeeping ─────────────────────────────────────────────────
    # 'epoch' stored is the *last completed* epoch (0-indexed).
    # We return epoch + 1 so the training loop resumes at the next epoch.
    last_epoch = checkpoint.get("epoch", -1)
    best_acc   = checkpoint.get("best_acc", 0.0)
    resume_from = last_epoch + 1

    print(f"[Checkpoint] Loaded ← {path}")
    print(f"             Resuming from epoch {resume_from}  "
          f"(best_acc so far = {best_acc:.4f})")
    return {"epoch": resume_from, "best_acc": best_acc}


# ---------------------------------------------------------------------------
# Model components  (unchanged from your original)
# ---------------------------------------------------------------------------

class LearnedFrequencyDecomposer(nn.Module):
    def __init__(self, image_size: int = 128, num_filters: int = 16):
        super().__init__()
        self.image_size = image_size
        self.num_filters = num_filters

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

        self.mix = nn.Sequential(
            nn.Conv2d(num_filters * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self._init_weights()

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

        fft = torch.fft.rfft2(x, norm='ortho')
        real = fft.real
        imag = fft.imag

        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        scale = mag.reshape(B, -1).std(dim=1).clamp(min=1e-4).view(B, 1, 1, 1)
        real = real / (scale + 1e-8)
        imag = imag / (scale + 1e-8)

        real_spatial = torch.fft.irfft2(
            torch.complex(real, torch.zeros_like(imag)),
            s=(self.image_size, self.image_size),
            norm='ortho'
        )
        imag_spatial = torch.fft.irfft2(
            torch.complex(torch.zeros_like(real), imag),
            s=(self.image_size, self.image_size),
            norm='ortho'
        )

        filtered_real = self.freq_conv_real(real_spatial)
        filtered_imag = self.freq_conv_imag(imag_spatial)

        combined = torch.cat([filtered_real, filtered_imag], dim=1)
        return self.mix(combined)


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

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(144, 36),
            nn.GELU(),
            nn.Linear(36, 144),
            nn.Sigmoid(),
        )

        self.compress = nn.Sequential(
            nn.Conv2d(144, 128, kernel_size=3, stride=2, padding=1, bias=False),
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
        s = self.period_small(freq_feat)
        m = self.period_mid(freq_feat)
        l = self.period_large(freq_feat)
        multi = torch.cat([s, m, l], dim=1)
        att = self.scale_attention(multi).view(multi.shape[0], 144, 1, 1)
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


class LightFrequencyBackbone(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.pool = nn.AdaptiveAvgPool2d(1)

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
        return self.pool(x).flatten(1)


# ---------------------------------------------------------------------------
# Main model
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
        self.artifact_detector = UpsamplingArtifactDetector(in_channels=64)
        self.backbone = LightFrequencyBackbone(in_channels=64)

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
        nn.init.zeros_(self.classifier.bias)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
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

    def forward(self, x: torch.Tensor, return_confidence: bool = False):
        freq_feat = self.freq_decomposer(x)
        artifact_feat = self.artifact_detector(freq_feat)
        backbone_feat = self.backbone(freq_feat)
        fused = self.fusion(torch.cat([artifact_feat, backbone_feat], dim=1))
        logits = self.classifier(fused)
        if self.temperature.item() != 1.0:
            logits = logits / self.temperature
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
    """
    Build FrequencyModelV5.

    Pass weights_path only if you want to load *weights only* (e.g. for
    inference).  For resuming training use load_checkpoint() directly so
    that the optimiser and epoch counter are also restored.
    """
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

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FrequencyModelV5 — Total params: {total:,} | Trainable: {trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Reference training loop  — copy / adapt into your own trainer
# ---------------------------------------------------------------------------

def train(
    model: FrequencyModelV5,
    train_loader,
    val_loader,
    *,
    device: torch.device,
    total_epochs: int = 100,
    lr: float = 1e-3,
    # Paths for the two checkpoints
    last_ckpt: str = "checkpoint_last.pth",   # saved every epoch
    best_ckpt: str = "checkpoint_best.pth",   # saved only when val_acc improves
):
    """
    Minimal but complete training loop with dual-checkpoint support.

    Two files are written to disk:
        last_ckpt — overwritten at the end of every epoch so resuming
                    always picks up from the most recent completed epoch.
        best_ckpt — overwritten only when validation accuracy improves,
                    giving you a clean best-model snapshot.

    To resume, just call train() again with the same paths — it detects
    the existing last_ckpt and continues from where it left off.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    # ── Restore previous run if a last checkpoint exists ──────────────────
    info = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
    start_epoch = info["epoch"]      # 0 on first run, N+1 after resuming
    best_acc    = info["best_acc"]

    if start_epoch >= total_epochs:
        print(f"Training already completed ({start_epoch} / {total_epochs} epochs).")
        return

    model.to(device)

    for epoch in range(start_epoch, total_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
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

        # ── Always save the "last" checkpoint so we can resume ─────────────
        save_checkpoint(
            last_ckpt, model, optimizer, scheduler,
            epoch=epoch,           # last *completed* epoch (0-indexed)
            best_acc=best_acc,
        )

        # ── Save the "best" checkpoint only when accuracy improves ─────────
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler,
                epoch=epoch,
                best_acc=best_acc,
            )
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
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "test.pth")
        opt  = torch.optim.AdamW(model.parameters())
        save_checkpoint(ckpt, model, opt, epoch=7, best_acc=0.923)
        info = load_checkpoint(ckpt, model, opt)
        assert info["epoch"] == 8,    f"Expected 8, got {info['epoch']}"
        assert abs(info["best_acc"] - 0.923) < 1e-6
        print(f"  Saved epoch=7  →  resumed from epoch={info['epoch']}  ✓")

    print("\nAll tests passed!")