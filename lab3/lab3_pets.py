import logging
import os
import subprocess
import time

import torch


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


def _reset_wandb():
    try:
        subprocess.run(["pkill", "-f", "wandb-core"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.5)
    os.environ.pop("WANDB_SERVICE", None)
    try:
        import wandb.sdk.wandb_setup as _wandb_setup

        _wandb_setup._WandbSetup._instance = None
    except Exception:
        pass


def _wandb_init_experiment(
    project,
    config,
    name=None,
    reset=True,
    watch_model=False,
    watch_log_freq=100,
    model=None,
):
    import wandb

    logging.getLogger("wandb").setLevel(logging.ERROR)
    if reset:
        _reset_wandb()
    kwargs = dict(
        project=project,
        config=config,
        reinit=True,
        settings=wandb.Settings(silent=True),
    )
    if name is not None:
        kwargs["name"] = name
    run = wandb.init(**kwargs)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    if watch_model and model is not None:
        wandb.watch(model, log="all", log_freq=watch_log_freq)
    return run


def get_dataloaders(
    root="./data",
    img_size=128,
    batch_size=32,
    train_fraction=0.85,
    num_workers=0,
    max_train_samples=None,
):
    t_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    t_test = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    full = OxfordIIITPet(
        root, split="trainval", target_types="category", download=True, transform=t_train
    )
    n = len(full)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(train_fraction * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if max_train_samples is not None:
        train_idx = train_idx[:max_train_samples]

    train_set = Subset(full, train_idx)
    test_full = OxfordIIITPet(
        root, split="trainval", target_types="category", download=True, transform=t_test
    )
    test_set = Subset(test_full, test_idx)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    num_classes = 37
    return train_loader, test_loader, num_classes


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def train_simple_cnn(
    epochs=15,
    lr=0.01,
    device=None,
    use_scheduler=True,
    use_wandb=False,
    max_train_samples=2000,
    weight_decay=0.0,
    wandb_reset=True,
    wandb_finish=True,
    wandb_watch=False,
    wandb_run_name=None,
):
    if device is None:
        device = _default_device()
    train_loader, test_loader, num_classes = get_dataloaders(
        max_train_samples=max_train_samples
    )
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        if use_scheduler
        else None
    )

    if use_wandb:
        import wandb

        if wandb.run is None:
            _wandb_init_experiment(
                "lab3-oxford-pets",
                dict(
                    lr=lr,
                    epochs=epochs,
                    use_scheduler=use_scheduler,
                    max_train_samples=max_train_samples,
                    weight_decay=weight_decay,
                    model="SimpleCNN",
                ),
                name=wandb_run_name,
                reset=wandb_reset,
                watch_model=wandb_watch,
                model=model,
            )
        else:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            if wandb_watch:
                wandb.watch(model, log="all", log_freq=100)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        if scheduler is not None:
            scheduler.step()
        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": ep,
                    "train/loss": tr_loss,
                    "test/loss": te_loss,
                    "train/accuracy": tr_acc,
                    "test/accuracy": te_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=ep,
            )

    if use_wandb and wandb_finish:
        import wandb

        wandb.finish()

    return model, history


def build_resnet18_finetune(num_classes=37, freeze_policy="last_block"):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    if freeze_policy == "head_only":
        pass
    elif freeze_policy == "last_block":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif freeze_policy == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(freeze_policy)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def finetune_train(
    epochs=15,
    lr_head=0.01,
    lr_backbone=0.001,
    freeze_policy="last_block",
    device=None,
    use_wandb=False,
    max_train_samples=None,
    weight_decay=0.0,
    wandb_reset=True,
    wandb_finish=True,
    wandb_watch=False,
    wandb_run_name=None,
):
    if device is None:
        device = _default_device()
    train_loader, test_loader, num_classes = get_dataloaders(
        max_train_samples=max_train_samples
    )
    model = build_resnet18_finetune(num_classes, freeze_policy=freeze_policy).to(device)
    criterion = nn.CrossEntropyLoss()
    head_ids = {id(p) for p in model.fc.parameters()}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    head_params = list(model.fc.parameters())
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    param_groups.append({"params": head_params, "lr": lr_head})
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if use_wandb:
        import wandb

        if wandb.run is None:
            _wandb_init_experiment(
                "lab3-oxford-pets",
                dict(
                    lr_head=lr_head,
                    lr_backbone=lr_backbone,
                    freeze_policy=freeze_policy,
                    epochs=epochs,
                    max_train_samples=max_train_samples,
                    weight_decay=weight_decay,
                    model="resnet18_finetune",
                ),
                name=wandb_run_name,
                reset=wandb_reset,
                watch_model=wandb_watch,
                model=model,
            )
        else:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            if wandb_watch:
                wandb.watch(model, log="all", log_freq=100)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        scheduler.step()
        if use_wandb:
            import wandb

            log_payload = {
                "epoch": ep,
                "train/loss": tr_loss,
                "test/loss": te_loss,
                "train/accuracy": tr_acc,
                "test/accuracy": te_acc,
                "lr_head": optimizer.param_groups[-1]["lr"],
            }
            if len(optimizer.param_groups) > 1:
                log_payload["lr_backbone"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_payload, step=ep)

    if use_wandb and wandb_finish:
        import wandb

        wandb.finish()

    return model, history


@torch.no_grad()
def extract_features(model, x, device):
    model.eval()
    x = x.to(device)
    feats = []
    handle = model.avgpool.register_forward_hook(lambda _m, _i, o: feats.append(o))
    _ = model(x)
    handle.remove()
    z = torch.flatten(feats[0], 1)
    return z


@torch.no_grad()
def nearest_neighbors_demo(model, train_loader, query_img_chw, device, k=5):
    model.eval()
    all_z, all_y = [], []
    for x, y in train_loader:
        x = x.to(device)
        z = extract_features(model, x, device)
        all_z.append(z.cpu())
        all_y.append(y.clone())
    Z = torch.cat(all_z, dim=0)
    Y = torch.cat(all_y, dim=0)
    q = extract_features(model, query_img_chw.unsqueeze(0), device).cpu()
    q = F.normalize(q, dim=1)
    Zn = F.normalize(Z, dim=1)
    sim = (Zn @ q.T).squeeze(1)
    top = torch.topk(sim, k=min(k, len(sim))).indices
    return Y[top].tolist(), sim[top].tolist()


def grad_cam_resnet(model, image_chw, target_class, device):
    model.eval()
    activations = []
    gradients = []

    def fwd_hook(_m, _i, o):
        activations.append(o)

    def bwd_hook(_m, _gi, go):
        gradients.append(go[0])

    hf = model.layer4.register_forward_hook(fwd_hook)
    hb = model.layer4.register_full_backward_hook(bwd_hook)
    x = image_chw.unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, target_class].backward()
    hf.remove()
    hb.remove()
    A = activations[0][0]
    G = gradients[0][0]
    alpha = G.mean(dim=(1, 2))
    cam = (alpha[:, None, None] * A).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(
        cam.view(1, 1, *cam.shape), size=(x.shape[2], x.shape[3]), mode="bilinear"
    )[0, 0]
    return cam.detach().cpu().numpy()


def occlusion_map(model, image_chw, target_class, device, patch_size=10, stride=5):
    model.eval()
    c, h, w = image_chw.shape
    base = image_chw.unsqueeze(0).to(device)
    with torch.no_grad():
        s0 = model(base)[0, target_class].item()
    heat = torch.zeros(h, w, device=device)
    counts = torch.zeros(h, w, device=device)
    for yy in range(0, h - patch_size + 1, stride):
        for xx in range(0, w - patch_size + 1, stride):
            oc = base.clone()
            oc[:, :, yy : yy + patch_size, xx : xx + patch_size] = 0
            with torch.no_grad():
                sc = model(oc)[0, target_class].item()
            delta = s0 - sc
            heat[yy : yy + patch_size, xx : xx + patch_size] += delta
            counts[yy : yy + patch_size, xx : xx + patch_size] += 1
    counts = torch.clamp(counts, min=1)
    heat = heat / counts
    return heat.cpu().numpy()


def top_k_activation_patches(
    model, loader, device, layer_name="layer3", channel=0, topk=25, patch_hw=32, max_batches=40
):
    model.eval()
    layer = getattr(model, layer_name)
    acts_buf = []

    def hook(_m, _i, o):
        acts_buf.append(o.detach())

    handle = layer.register_forward_hook(hook)
    candidates = []
    Hh = Ww = None
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        _ = model(x)
        A = acts_buf[-1][:, channel]
        if Hh is None:
            Hh, Ww = A.shape[1], A.shape[2]
        B = A.size(0)
        flat = A.reshape(B, -1)
        vals, idx = flat.max(dim=1)
        for b in range(B):
            v = vals[b].item()
            fi = idx[b].item()
            jj = fi // Ww
            ii = fi % Ww
            candidates.append((v, x[b].detach().cpu(), jj, ii))
        acts_buf.clear()
    handle.remove()
    candidates.sort(key=lambda t: -t[0])
    candidates = candidates[:topk]
    ih = iw = None
    patches = []
    for v, img, jj, ii in candidates:
        if ih is None:
            ih, iw = img.shape[1], img.shape[2]
        cy = int((jj + 0.5) / Hh * ih)
        cx = int((ii + 0.5) / Ww * iw)
        ph = pw = patch_hw // 2
        y0, x0 = max(0, cy - ph), max(0, cx - pw)
        y1, x1 = min(ih, cy + ph), min(iw, cx + pw)
        patch = img[:, y0:y1, x0:x1]
        if patch.shape[1] != patch_hw or patch.shape[2] != patch_hw:
            patch = F.interpolate(
                patch.unsqueeze(0), size=(patch_hw, patch_hw), mode="bilinear"
            )[0]
        patches.append(patch)
    return patches


def log_wandb_summary_table(rows, project="lab3-oxford-pets", run_name="summary-table"):
    import wandb

    if not rows:
        return
    logging.getLogger("wandb").setLevel(logging.ERROR)
    _reset_wandb()
    summary_run = wandb.init(
        project=project,
        name=run_name,
        reinit=True,
        settings=wandb.Settings(silent=True),
    )
    cols = list(rows[0].keys())
    table = wandb.Table(columns=cols)
    for r in rows:
        table.add_data(*[r[c] for c in cols])
    wandb.log({"results": table})
    summary_run.finish()


def run_wandb_sweep_train_fn():
    def train():
        import wandb

        c = wandb.config
        device = _default_device()
        freeze_policy = getattr(c, "freeze_policy", "last_block")
        max_samples = getattr(c, "max_train_samples", None)
        wd = float(getattr(c, "weight_decay", 0.0))
        watch = bool(getattr(c, "wandb_watch", False))
        _, history = finetune_train(
            epochs=int(c.epochs),
            lr_head=float(c.lr_head),
            lr_backbone=float(c.lr_backbone),
            freeze_policy=str(freeze_policy),
            device=device,
            use_wandb=True,
            max_train_samples=max_samples,
            weight_decay=wd,
            wandb_reset=False,
            wandb_finish=True,
            wandb_watch=watch,
        )
        te = history["test_acc"][-1]
        wandb.summary["final_test_accuracy"] = te
        return history

    return train
