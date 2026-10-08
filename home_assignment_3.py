"""
CS 5720 Neural Networks and Deep Learning, Fall 2026
Home Assignment 3 - Part II (Programming Questions)

Student: Jeffrey Agnitsch
Student ID: 7005146960

Usage:
    python home_assignment_3.py --q 1
    python home_assignment_3.py --q 2 --epochs 10

Q1: 2-D convolution implemented from scratch with NumPy (no built-in conv).
Q2: Transfer learning with a pretrained ResNet18 on the ants-vs-bees dataset.
    Experiment A freezes the whole convolutional base and trains only a new
    classifier. Experiment B also unfreezes the last convolutional block
    (layer4) and trains it together with the classifier.
"""

import argparse
import os
import time

import numpy as np

# Every run writes its text/plot results here so they can be committed with
# the repo and referenced from the README.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


class Tee:
    """Write everything printed to both the console and an output file."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.file)
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Q1: Convolution from scratch
# ============================================================================


def conv2d_scratch(x, w, stride=1, padding=0):
    """2-D convolution (cross-correlation, as used in CNNs) with plain loops.

    x        : 2-D NumPy array, the input image / feature map (H x W)
    w        : 2-D NumPy array, the filter / kernel (F x F)
    stride   : how many pixels the filter moves between positions
    padding  : number of zero rows/cols added on every side of the input

    Returns the output feature map as a 2-D NumPy array.

    Note: deep-learning frameworks implement "convolution" as cross-
    correlation (the filter is NOT flipped). That is what is done here. For
    this assignment's filter it makes no difference anyway, because the
    filter is symmetric under a 180-degree rotation.
    """
    # (a) zero-pad the input if requested
    if padding > 0:
        x = np.pad(x, pad_width=padding, mode="constant", constant_values=0)

    n_h, n_w = x.shape  # input height / width (after padding)
    f_h, f_w = w.shape  # filter height / width

    # Output size follows the standard formula  N_out = (N - F + 2P) / S + 1.
    # The 2P is already inside n_h / n_w because we padded above.
    out_h = (n_h - f_h) // stride + 1
    out_w = (n_w - f_w) // stride + 1
    out = np.zeros((out_h, out_w), dtype=x.dtype)

    # (b) slide the filter over the input ...
    for i in range(out_h):
        for j in range(out_w):
            # top-left corner of the current window in input coordinates
            r = i * stride
            c = j * stride
            window = x[r : r + f_h, c : c + f_w]
            # (c) ... and compute the dot product (element-wise multiply +
            # sum) at every valid location
            out[i, j] = np.sum(window * w)

    return out


def run_q1():
    log = Tee(os.path.join(OUT_DIR, "q1_output.txt"))

    # (a) store the input and the filter as NumPy arrays
    x = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    w = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.int64,
    )

    log("Input (5x5):")
    log(x)
    log("\nFilter (3x3):")
    log(w)

    # (b)(c)(d) stride 1, padding 0 - the configuration the question asks for
    out = conv2d_scratch(x, w, stride=1, padding=0)
    log("\nOutput feature map (stride=1, padding=0):")
    log(out)
    # (e) shape of the result
    log("Output shape:", out.shape)

    # (f) same input and filter, but stride 2, to show the effect directly
    out_s2 = conv2d_scratch(x, w, stride=2, padding=0)
    log("\nFor comparison, stride=2, padding=0:")
    log(out_s2)
    log("Output shape:", out_s2.shape)
    log(
        "\nStride 2 moves the filter two pixels at a time, so it visits only "
        "every other\nvalid position in each direction. Output size goes from "
        "(5-3)/1+1 = 3 to\n(5-3)/2+1 = 2 per side: the 3x3 map shrinks to 2x2 "
        "and keeps just the\ncorner values of the stride-1 result."
    )

    # sanity check against a one-line reference (explicit window sums) so the
    # from-scratch code is verified without calling any conv API
    ref = np.array([[np.sum(x[i : i + 3, j : j + 3] * w) for j in range(3)] for i in range(3)])
    assert np.array_equal(out, ref), "from-scratch result does not match reference"
    log.close()


# ============================================================================
# Q2: Transfer learning - freeze vs. fine-tune
# ============================================================================


def count_trainable(model):
    """Number of parameters that will receive gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


def build_model(mode, num_classes, device):
    """Load pretrained ResNet18 and configure which layers can train.

    mode == "frozen"    : Experiment A. Freeze every pretrained layer and
                          replace the final fully-connected layer with a new
                          one. Only that new layer trains.
    mode == "finetune"  : Experiment B. Same new classifier, but the last
                          convolutional block (layer4) is unfrozen too.
    """
    import torch.nn as nn
    from torchvision import models

    # ImageNet-pretrained weights downloaded (and cached) by torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything first. requires_grad=False means autograd will not
    # compute gradients for these tensors, so the optimizer cannot change them
    # and backprop stops early, which is also what makes it faster.
    for p in model.parameters():
        p.requires_grad = False

    if mode == "finetune":
        # Unfreeze the last residual block. ResNet18 is organised as
        # conv1 -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc, so
        # layer4 is the last convolutional block (two BasicBlocks, 512 ch).
        for p in model.layer4.parameters():
            p.requires_grad = True

    # Replace the 1000-way ImageNet classifier with a fresh num_classes-way
    # layer. New nn.Linear parameters default to requires_grad=True.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def train_one_experiment(name, mode, loaders, class_names, epochs, lr, device, log):
    """Train + validate one configuration and return the stats we must report."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model = build_model(mode, len(class_names), device)
    n_trainable = count_trainable(model)
    n_total = count_total(model)

    log("\n" + "=" * 70)
    log(name)
    log("=" * 70)
    log(f"Total parameters     : {n_total:,}")
    log(f"Trainable parameters : {n_trainable:,}  ({100 * n_trainable / n_total:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    # Only pass the parameters that can train - the frozen ones would just be
    # dead weight in the optimizer state.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    # ---- untimed warm-up ----
    # The first pass through the data pays one-time costs that have nothing to
    # do with freeze vs. fine-tune: spawning DataLoader workers, decoding every
    # JPEG into the OS file cache, and the GPU runtime picking/compiling
    # kernels for each new layer shape. Without this, whichever experiment
    # runs first looks slower than it is. One full untimed epoch of loading
    # plus one forward/backward on a batch removes that bias from the timing.
    model.train()
    first_batch = None
    for images, labels in loaders["train"]:
        if first_batch is None:
            first_batch = (images, labels)
    images, labels = first_batch[0].to(device), first_batch[1].to(device)
    criterion(model(images), labels).backward()
    optimizer.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---- pure-compute benchmark (no data loading) ----
    # ms per training step on a synthetic batch. This isolates the cost that
    # freezing is supposed to save: backprop through the frozen layers.
    x_syn = torch.randn(16, 3, 224, 224, device=device)
    y_syn = torch.randint(0, len(class_names), (16,), device=device)
    for _ in range(5):  # a few more warm-up iterations at this exact shape
        optimizer.zero_grad()
        criterion(model(x_syn), y_syn).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_bench = 20
    for _ in range(n_bench):
        optimizer.zero_grad()
        criterion(model(x_syn), y_syn).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_step = 1000 * (time.perf_counter() - t0) / n_bench
    log(f"Compute per step     : {ms_per_step:.1f} ms (forward+backward+update, batch 16, no data loading)")

    # Re-create the model so the benchmark updates do not leak into training.
    model = build_model(mode, len(class_names), device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    train_losses = []
    val_accs = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ---- training pass ----
        model.train()
        running_loss = 0.0
        n_seen = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        epoch_loss = running_loss / n_seen
        train_losses.append(epoch_loss)

        # ---- validation pass (no gradients) ----
        model.eval()
        correct = 0
        n_val = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                n_val += labels.size(0)
        val_acc = correct / n_val
        val_accs.append(val_acc)

        log(f"Epoch {epoch:2d}/{epochs}  train loss {epoch_loss:.4f}  val acc {100 * val_acc:.2f}%")

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start

    log(f"Training time        : {train_time:.1f} s")
    log(f"Final val accuracy   : {100 * val_accs[-1]:.2f}%")
    log(f"Best val accuracy    : {100 * max(val_accs):.2f}% (epoch {int(np.argmax(val_accs)) + 1})")

    return {
        "name": name,
        "trainable": n_trainable,
        "total": n_total,
        "time": train_time,
        "ms_per_step": ms_per_step,
        "final_acc": val_accs[-1],
        "best_acc": max(val_accs),
        "losses": train_losses,
        "val_accs": val_accs,
    }


DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"


def ensure_dataset(data_dir):
    """Download and unzip the ants-vs-bees dataset if it is not present.

    The zip (about 45 MB) expands to hymenoptera_data/{train,val}/{ants,bees}.
    data_dir is expected to point at the hymenoptera_data folder itself.
    """
    import urllib.request
    import zipfile

    if os.path.isdir(os.path.join(data_dir, "train")):
        return
    parent = os.path.dirname(os.path.abspath(data_dir))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "hymenoptera_data.zip")
    print(f"Dataset not found at {data_dir}; downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(parent)
    os.remove(zip_path)
    print("Download complete.")


def run_q2(data_dir, epochs, lr, batch_size, seed):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib

    matplotlib.use("Agg")  # headless backend: save the figure, don't show it
    import matplotlib.pyplot as plt

    ensure_dataset(data_dir)
    log = Tee(os.path.join(OUT_DIR, "q2_output.txt"))

    # Same seed for both experiments so the only difference is which layers
    # are allowed to train.
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log(f"Device: {device} ({dev_name})")
    log(f"PyTorch {torch.__version__}")

    # ImageNet normalisation constants - the pretrained weights expect inputs
    # scaled this way. Light augmentation on the training split only.
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                norm,
            ]
        ),
    }

    # ImageFolder infers the class from the sub-directory name:
    # data/hymenoptera_data/{train,val}/{ants,bees}/*.jpg
    sets = {s: datasets.ImageFolder(os.path.join(data_dir, s), tfms[s]) for s in ("train", "val")}
    loaders = {
        s: DataLoader(
            sets[s],
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=4,
            persistent_workers=True,  # keep workers alive between epochs
            pin_memory=(device.type == "cuda"),
        )
        for s in ("train", "val")
    }
    class_names = sets["train"].classes
    log(f"Classes: {class_names}")
    log(f"Train images: {len(sets['train'])}   Val images: {len(sets['val'])}")
    log(f"Epochs: {epochs}   Batch size: {batch_size}   LR: {lr}   Optimizer: SGD(momentum=0.9)")

    # ---- the two experiments, identical except for what is unfrozen ----
    results = [
        train_one_experiment(
            "Experiment A - Frozen Feature Extractor (train fc only)",
            "frozen", loaders, class_names, epochs, lr, device, log,
        ),
        train_one_experiment(
            "Experiment B - Fine-Tuned Network (train layer4 + fc)",
            "finetune", loaders, class_names, epochs, lr, device, log,
        ),
    ]

    # ---- (e) plot training loss for both runs on one figure ----
    plt.figure(figsize=(7, 4.5))
    for r in results:
        plt.plot(range(1, epochs + 1), r["losses"], marker="o", label=r["name"].split(" - ")[1])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("ResNet18 on ants vs. bees: frozen vs. fine-tuned")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "q2_training_loss.png")
    plt.savefig(plot_path, dpi=120)
    log(f"\nSaved loss plot to {plot_path}")

    # ---- (f) comparison table (markdown, ready to paste into README) ----
    log("\nComparison:\n")
    log(
        "| Method | Trainable Parameters | Training Time (10 epochs) | "
        "GPU Compute per Step | Accuracy (val, final) | Accuracy (val, best) |"
    )
    log("|---|---|---|---|---|---|")
    labels = ["Frozen Feature Extractor", "Fine-Tuned Network"]
    for lab, r in zip(labels, results):
        log(
            f"| {lab} | {r['trainable']:,} of {r['total']:,} | {r['time']:.1f} s | "
            f"{r['ms_per_step']:.1f} ms | {100 * r['final_acc']:.2f}% | {100 * r['best_acc']:.2f}% |"
        )
    log.close()


# ============================================================================
# entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 5720 Home Assignment 3 - Part II")
    parser.add_argument("--q", type=int, required=True, choices=[1, 2], help="question to run")
    parser.add_argument("--epochs", type=int, default=10, help="Q2: epochs for BOTH experiments")
    parser.add_argument("--lr", type=float, default=0.001, help="Q2: SGD learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Q2: batch size")
    parser.add_argument("--seed", type=int, default=42, help="Q2: random seed")
    parser.add_argument(
        "--data_dir",
        default=os.path.join("data", "hymenoptera_data"),
        help="Q2: folder containing train/ and val/ sub-folders",
    )
    args = parser.parse_args()

    if args.q == 1:
        run_q1()
    else:
        run_q2(args.data_dir, args.epochs, args.lr, args.batch_size, args.seed)
