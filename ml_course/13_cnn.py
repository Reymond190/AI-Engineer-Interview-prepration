"""
13 - Convolutional networks: what convolution buys you
======================================================

Interview topics
----------------
- A conv layer slides a small kernel (k x k x C_in) over the image: the same
  weights at every position (WEIGHT SHARING) and each output looks only at a
  k x k neighbourhood (LOCALITY). Params = k*k*C_in*C_out + C_out,
  independent of image size; a dense layer on the same image needs
  H*W*C_in*H'*W'*C_out.
- Output size: floor((H + 2p - k) / s) + 1. "Same" padding p = (k-1)/2 for
  stride 1.
- Equivariance vs invariance: convolution is translation EQUIVARIANT (shift
  the input -> the feature map shifts). Pooling (max/avg, esp. global average
  pooling) makes the final prediction approximately translation INVARIANT.
  An MLP has neither: a pattern it learned at the left is a different input
  at the right (section 3 measures this).
- Receptive field grows with depth: r_l = r_{l-1} + (k-1) * prod(strides
  before l). Stacking two 3x3 convs sees 5x5 with fewer params than one
  5x5 (VGG insight) and an extra non-linearity.
- 1x1 convs mix channels (bottlenecks, ResNet/Inception). Depthwise-separable
  convs (MobileNet) = depthwise k x k per channel + 1x1 pointwise: ~k^2 fewer
  FLOPs.
- Implementation: im2col (unfold patches into columns) turns convolution
  into one big matrix multiply -> GPUs/cuDNN; "convolution" in DL is really
  cross-correlation (no kernel flip).
- History in one line: LeNet -> AlexNet (ReLU, dropout, GPU) -> VGG (3x3
  stacks) -> ResNet (residuals) -> EfficientNet/ConvNeXt; ViTs replace convs
  with patch embeddings + attention and need more data or pretraining because
  they lack these built-in priors.

Run:  python ml_course/13_cnn.py [--quick]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, table

SHAPES = {  # 4 classes of 5x5 glyphs
    "plus": ["..#..", "..#..", "#####", "..#..", "..#.."],
    "box": ["#####", "#...#", "#...#", "#...#", "#####"],
    "x": ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "ell": ["#....", "#....", "#....", "#....", "#####"],
}
GLYPHS = np.stack([np.array([[c == "#" for c in row] for row in g], dtype=np.float32) for g in SHAPES.values()])


def make_images(n: int, size: int, cols: tuple[int, int], rng: np.random.Generator, noise: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Glyph pasted at a random row and a random column in [cols[0], cols[1]); plus Gaussian noise."""
    x = rng.normal(0, noise, size=(n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, len(GLYPHS), n)
    for i in range(n):
        r, c = rng.integers(0, size - 4), rng.integers(cols[0], cols[1])
        x[i, 0, r:r + 5, c:c + 5] += GLYPHS[y[i]]
    return x, y


def conv2d_loops(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Direct definition: out[n, o, i, j] = sum_{c,u,v} x[n, c, i*s+u, j*s+v] * w[o, c, u, v] + b[o]."""
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    n, _, h, wd = x.shape
    co, _, k, _ = w.shape
    ho, wo = (h - k) // stride + 1, (wd - k) // stride + 1
    out = np.zeros((n, co, ho, wo), dtype=np.float64)
    for i in range(ho):
        for j in range(wo):
            patch = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def conv2d_im2col(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int = 1, pad: int = 0) -> torch.Tensor:
    """im2col: unfold every k x k patch into a column, then ONE matmul with the flattened kernels."""
    n, _, h, wd = x.shape
    co, _, k, _ = w.shape
    cols = F.unfold(x, k, padding=pad, stride=stride)            # (n, C_in*k*k, L)
    out = w.view(co, -1) @ cols + b.view(1, co, 1)               # (n, C_out, L)
    ho, wo = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    return out.view(n, co, ho, wo)


class SmallCNN(nn.Module):
    def __init__(self, ch: int = 16, global_pool: bool = True, size: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(ch, 2 * ch, 3, padding=1), nn.ReLU())
        self.global_pool = global_pool
        self.head = nn.Linear(2 * ch if global_pool else 2 * ch * (size // 2) ** 2, len(GLYPHS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.amax(dim=(2, 3)) if self.global_pool else h.flatten(1)  # global MAX pool -> position-invariant
        return self.head(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    rng = np.random.default_rng(args.seed)

    section("1. Convolution three ways: loops (definition), im2col + matmul, F.conv2d")
    x = rng.normal(size=(2, 3, 9, 9))
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=4)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=2, padding=1).numpy()
    loops = conv2d_loops(x, w, b, stride=2, pad=1)
    im2col = conv2d_im2col(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=2, pad=1).numpy()
    kv("conv-parity", loops_vs_torch=float(np.abs(loops - ref).max()), im2col_vs_torch=float(np.abs(im2col - ref).max()),
       out_hw=ref.shape[-1], formula=(9 + 2 * 1 - 3) // 2 + 1)

    section("2. Parameter counts: conv vs dense on a 32x32x3 image -> 64 channels")
    conv = nn.Conv2d(3, 64, 3, padding=1)
    dense_params = (32 * 32 * 3) * (32 * 32 * 64) + 32 * 32 * 64
    sep = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 128, 1))
    full = nn.Conv2d(64, 128, 3, padding=1)
    table([{"layer": "Conv2d(3, 64, 3x3)", "params": n_params(conv)},
           {"layer": "Dense 32*32*3 -> 32*32*64", "params": dense_params},
           {"layer": "Conv2d(64, 128, 3x3)", "params": n_params(full)},
           {"layer": "depthwise 3x3 + pointwise 1x1 (64->128)", "params": n_params(sep)}])
    kv("params", conv=n_params(conv), dense=dense_params, separable_ratio=n_params(full) / n_params(sep))
    rf, jump = 1, 1
    for name, k, s in [("conv3", 3, 1), ("conv3", 3, 1), ("pool2", 2, 2), ("conv3", 3, 1)]:
        rf += (k - 1) * jump
        jump *= s
        print(f"  after {name}: receptive field {rf}x{rf}")
    kv("receptive-field", small_cnn=rf)

    section("3. Translation: train with glyphs on the LEFT, test with glyphs on the RIGHT")
    size = 24
    n_train = 2000 if args.quick else 6000
    xtr, ytr = make_images(n_train, size, (0, 8), rng)            # glyph columns 0..7 (left side)
    xin, yin = make_images(1000, size, (0, 8), rng)               # same region (in-distribution)
    xsh, ysh = make_images(1000, size, (12, size - 4), rng)       # columns 12..19: never seen
    t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    xtr, ytr, xin, yin, xsh, ysh = map(t, (xtr, ytr, xin, yin, xsh, ysh))
    models = {
        "MLP (2 hidden x 256)": nn.Sequential(nn.Flatten(), nn.Linear(size * size, 256), nn.ReLU(), nn.Linear(256, 256),
                                              nn.ReLU(), nn.Linear(256, len(GLYPHS))),
        "CNN + flatten head": SmallCNN(global_pool=False, size=size),
        "CNN + global max pool": SmallCNN(global_pool=True, size=size),
    }
    rows = []
    epochs = 8 if args.quick else 20
    for name, m in models.items():
        torch.manual_seed(args.seed)
        m = m.to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(xtr), 64):
                idx = perm[i:i + 64]
                loss = F.cross_entropy(m(xtr[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            acc_in = (m(xin).argmax(1) == yin).float().mean().item()
            acc_sh = (m(xsh).argmax(1) == ysh).float().mean().item()
        rows.append({"model": name, "params": n_params(m), "test acc (left, seen region)": acc_in,
                     "test acc (right, unseen region)": acc_sh})
    table(rows)
    kv("translation", mlp_shifted=rows[0]["test acc (right, unseen region)"], cnn_flat_shifted=rows[1]["test acc (right, unseen region)"],
       cnn_pool_shifted=rows[2]["test acc (right, unseen region)"], cnn_pool_params=rows[2]["params"], mlp_params=rows[0]["params"])
    print("  The MLP has 30x the parameters and barely learns even the seen region: it must learn each glyph")
    print("  separately at every position (no weight sharing -> poor sample efficiency). The CNN with a flatten")
    print("  head learns the seen region but ties each feature to a position, so it fails on the unseen one.")
    print("  Only weight sharing + global pooling transfers to positions never seen in training.")
    print("  (Chance = 0.25. Data augmentation with random shifts is the usual way to get this for other models.)")


if __name__ == "__main__":
    main()
