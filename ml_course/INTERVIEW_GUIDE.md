# ML & DL fundamentals: interview guide

Short answers to know from memory. Numbers in parentheses point to the lesson
that demonstrates the topic. Where a number is **measured here**, it comes
from a `--quick` run of that lesson (seed 0); rerun to see it yourself.

---

## 1. Data, splits and evaluation (00, 07)

**Why train / validation / test?**
Train fits the weights. Validation chooses everything else: hyperparameters,
model, threshold, early-stopping epoch. Test is used **once** to report. Any
choice made by looking at a set leaks it into training. **Measured (00):** with
coin-flip labels, the best of 50 models picked by test score showed 0.625 test
accuracy, and 0.525 on fresh data.

**k-fold CV: why and when?**
Every point is validated once, and the averaged estimate is much less noisy
than one split. **Measured (00):** breast cancer, std over seeds 0.010 for a
single 80/20 split vs 0.002 for the 5-fold mean. Use stratified folds for
classification, GroupKFold when entities repeat, TimeSeriesSplit for time
order, and nested CV if you both tune and report on the same data.

**What is data leakage? Give examples.**
Information that wouldn't be available at prediction time reaches training.
**Measured (00):**

- Feature selection on all data before CV: 0.88 CV accuracy on pure noise, vs 0.47 inside a Pipeline.
- Same patient in train and validation folds: KFold 0.995, GroupKFold 0.53.
- Shuffled CV on a random walk: R² 0.995, walk-forward 0.11.
- Others: target leakage (a feature known only after the label), duplicates
across splits, scalers/imputers/encoders fitted on everything.

The fix is to put every fitted step in a Pipeline that is fit inside each fold.

**Which metric?**
It depends on the cost of each error. Under imbalance accuracy lies.
**Measured (07):** at 2% positives, "always negative" scores 0.98 accuracy with
0 recall.

- **Precision** = TP/(TP+FP); **recall** = TP/(TP+FN); **F1** is their harmonic mean.
- **ROC-AUC** is the probability that a random positive outranks a random
negative. It ignores the class ratio.
- **PR-AUC** shows what happens to precision when positives are rare.
**Measured (07):** with the same scorer, as positives go from 50% to 0.5%,
ROC-AUC stays at 0.92 and PR-AUC falls from 0.92 to 0.17.

**Imbalanced data: what do you do?**
In order: the right metric, then tune the threshold on validation (07: F1
0.16 → 0.33 by moving the threshold from 0.5 to 0.17), then class weights (07:
recall 0.14 → 0.83, precision 0.57 → 0.10), then resampling **inside the
training fold only**, then more positives. Class weights and resampling distort
probabilities, so recalibrate afterwards if you need real probabilities.

**What is calibration and how do you fix it?**
Of all cases predicted at 0.8, about 80% should be positive. Measure it with
ECE, Brier score or a reliability curve. **Measured (07):** with duplicated
(correlated) features, naive Bayes has ECE 0.20 vs 0.008 for logistic
regression, because NB multiplies the same evidence five times. Isotonic
regression brought it to 0.012, and AUC was unchanged. Platt (sigmoid) scaling
is better with little data; isotonic is more flexible.

---

## 2. Linear models and optimisation (01, 02)

**Linear regression: how is it solved?**

- Minimise MSE. The closed form is w = (XᵀX)⁻¹Xᵀy, but in practice use
QR/SVD (lstsq), because forming XᵀX squares the condition number.
- Gradient descent is used when n or d is huge.
- **Measured (01):** with near-duplicate features (cond 3e12), coefficients vary
by ±33,000 across bootstrap resamples while their sum stays at 3.0.
Predictions are fine; the individual coefficients mean nothing. Ridge or
dropping one of the features fixes it.

**Why cross-entropy, not MSE, for classification?    **  
CE is the negative log-likelihood of a Bernoulli/categorical, and for  
logistic regression it is convex. Its gradient with respect to the logit is  
p − y. MSE through a sigmoid has gradient 2(p−y)·p(1−p), which vanishes  
exactly when the model is confidently wrong. **Measured (01):** at z = −10 with
y = 1, the CE gradient is −1.0 and the MSE gradient is −9e-5.

**Softmax numerical stability?**
Subtract the max logit before exp (the log-sum-exp trick); the result doesn't
change. **Measured (01):** the naive softmax of [1000, 999, 0] gives NaN.
Frameworks fuse log_softmax with NLL for the same reason.

**What does the learning rate do on a quadratic?**

- GD converges iff lr < 2/L (L = largest curvature). **Measured (02):** lr·L = 1.99
converges, 2.01 blows up.
- Steps needed grow with the condition number κ = L/μ. **Measured (02):** κ = 1000
took 6160 GD steps vs 987 with tuned momentum (≈√κ scaling).

**Why scale features?**
Different units make the loss a long thin valley, with huge κ. **Measured (02):**
metres + millimetres gave κ = 8.6e5, and GD hadn't converged after 100k steps.
Standardised, κ = 1.01 and it converged in 78 steps. This matters for GD-trained
models, kNN, SVM, PCA and regularised models; trees don't care.

**Momentum, RMSProp, Adam in one line each?**

- **Momentum** averages gradients, which damps zig-zag and speeds you along valleys.
- **RMSProp** divides each coordinate by a running RMS of its gradient, which
gives per-parameter step sizes.
- **Adam** = momentum + RMSProp + bias correction (both moments start at 0).

**Measured (02, Rosenbrock, 3000 steps):** GD at lr 1e-3 reached f = 0.049 and
Adam f = 3e-11. RMSProp at constant lr ended bouncing between two values
(0.0223 and 0.0226), because its step is ~lr whatever the gradient size, so
you need lr decay.

**SGD vs full-batch GD?**
A minibatch gradient is unbiased but noisy. With a constant lr the loss stalls
at a noise floor ∝ lr/batch. **Measured (02):** excess loss 0.39 (batch 1,
constant), 0.033 (batch 1, decayed), 0.009 (batch 32, constant). Noise also
regularises (tends toward flat minima) and helps escape saddle points.

---

## 3. Bias, variance, regularisation (03)

**What are bias and variance?**
Expected error = bias² + variance + noise. Bias is the error of the
*average* model (too simple). Variance is how much the model changes with the
training set (too flexible). **Measured (03), sin curve with 30 points and
noise variance 0.09, 100 refits:**

| degree | bias² | variance |
|---|---|---|
| 1 | 0.175 | 0.022 |
| 3 | 0.004 | 0.016 |
| 12 | huge, from Runge-style blow-ups at the edges | 2.5e4 |

**How do you diagnose it?**

- High train and high val error (small gap) is bias: use a bigger model, better
features, less regularisation.
- Low train and high val error (big gap) is variance: get more data,
regularise, simplify, ensemble, or stop early.
- Learning curves: more data fixes variance, not bias. **Measured (03):** at
n = 1000, degree 1 is stuck at 0.28 test MSE and degree 12 reaches 0.09 (the
noise floor).

**L1 vs L2?**

- **L2** shrinks all weights smoothly, is never exactly 0, and handles correlated
features well (Gaussian prior). It also makes XᵀX invertible.
- **L1** gives exact zeros, so it does feature selection (Laplace prior). Its
proximal step is soft-thresholding.
- **Measured (03):** 5 real features out of 50. Lasso zeroed 45 weights and kept
all 5 real ones; ridge zeroed none.
- Always scale features first.

**Regularisation strength?**
Too little and variance wins; too much and bias wins. **Measured (03):** degree
12 test MSE was 52 at α = 0, 0.17 at α = 1, and 0.26 at α = 10. Tune it on validation.

---

## 4. Trees and ensembles (04)

**How does a decision tree split?**
Greedily, on the feature and threshold that most reduce the size-weighted
impurity. Gini = 1 − Σp², entropy = −Σp log p. [50, 50] has Gini 0.5 and 1
bit; [99, 1] has Gini 0.02. Regression trees use variance. Unlimited depth
memorises: **measured (04)** train accuracy 1.0 vs test 0.92. Control it with
max_depth, min_samples_leaf or ccp_alpha.

**Random forest vs bagging vs a single tree?**

- Bagging trains trees on bootstrap samples (~63% unique rows) and averages them.
- A random forest also samples features at each split, which decorrelates the
trees. The variance of an average is ρσ² + (1−ρ)σ²/B, so correlation is
what limits it.
- **Measured (04)** (noisy labels, extra noise features): single tree 0.81,
bagging 0.92 (tree correlation 0.48), RF 0.94 (correlation 0.45).
- Out-of-bag (OOB) rows give free validation. Forests mostly reduce variance,
and more trees never overfit.

**Gradient boosting?**

- Trees are added sequentially, each fitting the negative gradient (for MSE,
the residuals) of the current ensemble and scaled by the learning rate.
Boosting mostly reduces bias.
- It can overfit, so use early stopping.
- **Measured (04):** lr 1.0 hit its best test MSE at round 10 and then overfit
(4.7 → 5.8). lr 0.1 reached 1.92, matching sklearn's GBM (1.93).
- XGBoost, LightGBM and HistGradientBoosting add Newton steps, histogram
binning, L1/L2 penalties on leaves and native missing values. A tuned GBM
is still the tabular default.

**Feature importance caveat?**
Impurity importance is computed on training data and is biased toward
high-cardinality features. **Measured (04):** a random ID column got impurity
importance 0.016 but permutation importance 0.0005. Use permutation importance
on held-out data, or SHAP.

---

## 5. SVM, kNN, curse of dimensionality (05)

**SVM in one breath?**
Maximise the margin: minimise ½‖w‖² + C·Σ hinge(y·f(x)). Only points with
margin < 1 (the support vectors) shape the solution. **Measured (05):** 18 of
398 training points were on or inside the margin. Small C gives a wide
margin and many support vectors (188 at C = 0.01, underfit); large C
gives few (55). Pegasos subgradient descent matched LinearSVC (0.953 both).

**Kernel trick?**
The dual only needs dot products, so replace them with k(x, z) = φ(x)·φ(z).
**Measured (05):** (x·z + 1)² equals the dot product of explicit 10-d features
to 4e-16. On circles a linear SVM gets 0.62 and RBF 1.00. RBF gamma is the
inverse width: γ = 1000 gave train 1.0 and test 0.58. Kernel SVMs cost
O(n²)–O(n³) in samples.

**kNN pitfalls?**

- **Scaling:** **measured (05)** one useless feature in large units dropped kNN
from 0.88 to 0.56.
- **Curse of dimensionality:** **measured (05)** nearest/farthest distance ratio
is 0.01 in 2-d and 0.91 in 1000-d, and accuracy fell from 0.91 to 0.58 with
200 noise dims added.
- k trades bias for variance: k = 1 gives train 1.0.

---

## 6. Unsupervised (06)

**k-means?**
Lloyd's algorithm alternates assigning points and taking means. It decreases
inertia monotonically, but only to a **local** minimum. **Measured (06):** over
single runs, random init had median inertia 2064 vs 1176 for k-means++, and
even k-means++ had a bad run (2386 vs 1783), hence n_init. It assumes convex,
similar-size clusters. **Measured (06):** ARI 0.26 on moons vs 1.0 for DBSCAN.

**Choosing k?**
Inertia always falls with k, so look for the elbow, or take the max
silhouette. **Measured (06):** silhouette picked k = 4 = true k.

**PCA?**
Centre the data and take the SVD. The components are the right singular
vectors, and the explained variance is S²/(n−1). It's unsupervised: high
variance ≠ predictive. **Measured (06):** digits need 29 of 64 components for
95% variance. Unscaled wine data: one feature (proline, std 314) is 99.8% of
PC1, so standardise first. t-SNE/UMAP are for visualisation only.

---

## 7. Backprop and training neural nets (08, 09)

**How does backprop work?**
The chain rule in reverse topological order. Each node passes grad × local
derivative to its inputs, and grads from several uses add up. One backward
pass costs about 2× a forward pass. **Measured (08):** a 60-line scalar
autograd engine matched torch to 2e-16. Hand-written numpy MLP backprop passed
a finite-difference check (relative error 2e-8).

- For y = xW + b: dW = xᵀdy, db = Σdy, dx = dy·Wᵀ.
- Softmax + CE: dlogits = (p − onehot)/n.

**What uses memory in training?**
Weights, gradients, optimizer state, and the activations saved for backward,
which grow with batch × sequence. **Measured (08):** a 32 MB MLP saved 240 MB of
activations at batch 4096. Hence gradient checkpointing.

**PyTorch gotchas?**

- `.grad` accumulates: **measured (08)** three backward() calls gave 3× the gradient.
- `no_grad` doesn't build a graph.
- `detach()` cuts the graph.
- `model.eval()` switches dropout and BatchNorm: **measured (09)** a
dropout-0.5 net scored 0.998 in eval mode and 0.973 ± noise in train mode.

**How do you debug a net that won't train?**

1. Check the loss at init is ≈ ln(C). **Measured (09):** 1.11 vs ln 3 = 1.10;

   initialising the last layer with std 10 gave 7.2.
2. Overfit a single batch to ~0. **Measured:** 3e-5. A big net can memorise even

   shuffled labels, so this only proves something when it **fails**.
3. Scale up and watch train and val curves together.
4. Add regularisation once you overfit.
5. If the loss is NaN: lower the lr, clip gradients, check inputs for inf/NaN,

   and check for log(0).

**Regularising a neural net?**
Weight decay, dropout, data augmentation, early stopping with best-checkpoint
restore, a smaller model, label smoothing. **Measured (09)** (150 points, 20%
wrong labels, 400 epochs): the unregularised net's val loss bottomed at 0.16
and ended at 0.49 as it memorised noise. Dropout ended at 0.32, early stopping
stopped at epoch 139, and restoring the best-val checkpoint gave every setup
its best test accuracy (0.91–0.93 vs 0.82–0.87).

---

## 8. Making deep nets trainable (10, 11)

**Vanishing / exploding gradients?**
Each layer scales the backward signal by roughly weight scale × activation
slope; across depth this compounds exponentially. **Measured (10), 20 layers:**
sigmoid's gradient at layer 1 is 1e-12 of layer 20's, because its slope is
≤ 0.25. Tanh with std-1.0 init is chaotic: the gradient grows 3e7× going back.
Fixes: ReLU-family activations, proper init, normalisation, residuals, and
clipping (**measured:** a bad init gave grad norm 3e18, clipped to 1.0).

**Xavier vs He init?**
Both keep activation variance constant across layers. Xavier: Var = 2/(fan_in

+ fan_out), for tanh/linear. He: Var = 2/fan_in, for ReLU, which zeroes half
its inputs. **Measured (10):** ReLU + Xavier std at layer 20 = 0.0005; ReLU + He
= 0.49. Constant init never breaks symmetry: **measured** 16 hidden units
stayed identical through 200 steps.

**Why do residual connections work?**
y = x + F(x) has Jacobian I + F′, so there is an identity path for the
gradient. **Measured (10):** at depth 100, plain ReLU + He stayed at chance
(0.33) while residual + LayerNorm reached 1.0.

**BatchNorm vs LayerNorm vs RMSNorm?**

- **BatchNorm** normalises each feature over the batch, using running stats at
eval. A sample's output depends on its batch-mates (**measured (11)**:
output diff 2.4 vs 0 for LN), it can't train on batch 1, and it is bad with
tiny batches (**measured** batch 2: BN 0.56 vs LN 0.99).
- **LayerNorm** normalises each sample over its features, so train and eval are
identical. It's the transformer default.
- **RMSNorm** drops the mean and β (Llama/Qwen).
- **Measured (11), placement matters:** in a plain post-norm MLP, LN/RMSNorm
only trained at lr 1e-3 while BN trained up to 1e-1. In a pre-norm residual
stack all three trained at lr 0.1 and the no-norm net diverged.
- Caveat: LN(A·x) ignores the scale of x per sample (**measured:**
LN(Ax) ≈ LN(A·10x)).

**Pre-LN vs post-LN?**
Pre-LN (x + F(LN(x))) keeps a clean residual path and trains stably. Post-LN
(the original transformer) needs careful warmup.

---

## 9. Optimizers and schedules in practice (12)

**How do you pick a learning rate?**
Run an LR range test: increase lr exponentially, find where the loss is lowest
or starts to blow up, and take ~1/10 of that. **Measured (12):** blow-up at
0.15, suggested 5e-3, best test accuracy 0.978. Adam at lr 0.1 dropped to 0.75.

**SGD vs Adam?**
Adam converges faster with little tuning and handles very different gradient
scales across parameters; tuned SGD + momentum can generalise as well on
CNNs. **Measured (12), 600 steps:** SGD 0.81, momentum 0.94, Nesterov 0.96,
Adam 0.97.

**Adam + L2 vs AdamW?**
In Adam + L2 the decay term is divided by √v along with the gradient, so the
decay a weight gets depends on its gradient history. AdamW decouples it:
w ← w − lr·(adam_step + λw). **Measured (12):** weights on always-blank pixels
(no gradient) kept 4.5e-10 of their size under Adam + L2 (fully wiped out)
and 0.55 under AdamW. Note the scale: AdamW at wd 1e-2, lr 1e-3 for 600 steps
shrinks weights only ~0.6%. Don't decay biases or norm gains.

**Warmup and decay?**
Warmup protects the early steps, while Adam's moment estimates are still
noisy. Decay (cosine, linear, step, or warmup-stable-decay) removes the late
noise floor. **Measured (12):** at peak lr 3e-2, constant lr ended at 0.90 test
accuracy vs 0.97 with warmup + cosine. At a low peak lr the schedules were
within noise of each other.

**Batch size and learning rate?**
Linear scaling rule: k× the batch → k× the lr (SGD), with warmup; for Adam use
roughly √k. It breaks down past the critical batch size. **Measured (12):**
batch 256 with lr fixed at 0.02 gave test loss 0.24; scaled to 0.08 it gave 0.15.

---

## 10. Architectures (13, 14, 15)

**What does a CNN buy you?**
Locality and weight sharing, so the parameter count doesn't depend on image
size. **Measured (13):** a 3×3 conv from 3 to 64 channels has 1.8k params; a
dense layer for the same 32×32 map has 201M. Convolution is translation
*equivariant*, and global pooling makes the model *invariant*. **Measured (13),
glyphs trained only on the left half:**

| model | params | acc, seen region | acc, unseen region |
|---|---|---|---|
| MLP | 214k | 0.38 | 0.26 |
| CNN + flatten head | 25k | 0.90 | 0.24 |
| CNN + global max pool | 7k | 0.998 | 0.999 |

- Output size = ⌊(H + 2p − k)/s⌋ + 1.
- Two 3×3 convs see 5×5 with fewer params.
- Depthwise-separable convs: **measured** 8.2× fewer params than a full 3×3.
- im2col turns convolution into one matmul.

**RNN vs LSTM vs GRU?**
BPTT multiplies a Jacobian per step, so vanilla RNN gradients vanish. The
LSTM's additive cell update (c = f·c + i·g, with forget-gate bias 1) keeps
a gradient path. The GRU has 2 gates instead of 3 and fewer params.
**Measured (14):** at length 50 the share of gradient reaching token 0 was
5e-16 (RNN) vs 8e-5 (LSTM). Recall of the first token, 1500 steps: at length 20,
RNN 0.12, LSTM/GRU 1.00; at length 30 all cells failed. Gating extends memory;
it doesn't remove the limit.

**Attention?**
softmax(QKᵀ/√d_k + mask)·V.

- **Why √d_k:** dot products of random vectors have variance d_k, so without
scaling the softmax saturates and its gradient vanishes. **Measured (15):**
at d = 1024 unscaled, the mean max weight was 0.96 and the gradient scale
0.06; scaled, 0.17 and 0.93 for every d.
- **Causal mask:** −inf above the diagonal.
- **Multi-head:** h heads of size d/h with the same FLOPs as one big head;
GQA/MQA share K/V between heads.
- **Positions:** attention is permutation-equivariant (**measured:** diff 2e-7),
so order has to be added (learned, sinusoidal, RoPE, ALiBi).
- **Measured (15):** a 2-layer transformer recalled the first of 50 tokens with
accuracy 1.00 with positions and 0.19 without; the RNNs in 14 were at chance.

**Transformer block and cost?**
Pre-LN: x + Attn(LN(x)), then x + MLP(LN(x)), with a 4× MLP or SwiGLU.
Attention is O(n²·d) in sequence length. FlashAttention computes the same math
in tiles without storing the n×n matrix. The KV cache stores the past K and V
so each new token costs O(n·d). **Measured (15):** cached incremental decoding
matched full recompute (diff 6e-7) with 32× fewer K/V projection FLOPs at
n = 64. Cache size = 2 × layers × kv_heads × head_dim × bytes per token:
**128 KiB/token for an 8B GQA model, 4 GiB for one 32k sequence**.
