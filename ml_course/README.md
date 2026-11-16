# ML & DL fundamentals course (runnable)

The runnable companion to [`../ml_fundamentals.md`](../ml_fundamentals.md):
every classic interview topic as a short script that **measures the effect**
instead of asserting it. Leakage inflates a score, unscaled features slow
gradient descent, BatchNorm breaks at batch size 2, an RNN forgets after 20
steps. Same style as `langgraph_course/` and `finetune_course/`: each script
opens with an **"Interview topics"** docstring, then prints `[tag] key=value`
result lines that `run_all.py` checks. `INTERVIEW_GUIDE.md` holds the short
answers, with the numbers measured here.

Algorithms are written **from scratch** (numpy / plain torch) and checked
against sklearn or torch, so you know both how they work and how to call them.

## Setup

Runs in the WSL conda env `ft` (numpy, scikit-learn, scipy, torch), which
`setup/05_setup_finetune_env.sh` creates. The models are tiny, so **vLLM can
stay up**, and every lesson also runs on CPU. Data is synthetic or bundled
with sklearn (breast_cancer, digits, diabetes, wine), so nothing is
downloaded.

```bash
# WSL, project root
conda activate ft
python ml_course/00_splits_cv_leakage.py            # one lesson (add --quick for fewer steps)
python ml_course/run_all.py                         # all lessons, --quick, outcome checks (~6 min)
python ml_course/run_all.py 09 11                   # selected
python ml_course/run_all.py --full --save results_ml_full.json
```

Logs go to `ml_course/outputs/logs/` (gitignored).

## Lessons

### Part 1: classic ML (numpy + scikit-learn)
| # | Script | Core idea (what it measures) |
|---|--------|-----------|
| 00 | `00_splits_cv_leakage.py` | Selecting on test is optimistic; feature-selection, group and time leakage; one split vs k-fold variance |
| 01 | `01_linear_logistic.py` | Normal equation / lstsq / GD; collinearity; why CE not MSE; logistic GD + gradcheck; stable softmax |
| 02 | `02_gradient_descent.py` | lr < 2/L; steps ∝ condition number; feature scaling; GD vs momentum vs RMSProp vs Adam; SGD noise floor |
| 03 | `03_bias_variance_regularization.py` | U-curve; bias² and variance measured over 100s of refits; ridge; learning curves; L1 vs L2 sparsity |
| 04 | `04_trees_forests_boosting.py` | Gini/entropy; CART from scratch; bagging vs RF decorrelation, OOB; boosting on residuals; importance pitfalls |
| 05 | `05_svm_knn.py` | Pegasos (hinge) SVM; C and support vectors; kernel trick identity; RBF gamma; kNN scaling; curse of dimensionality |
| 06 | `06_kmeans_pca.py` | k-means + k-means++ from scratch; elbow/silhouette; DBSCAN on moons; PCA via SVD; PCA without scaling |
| 07 | `07_metrics_imbalance.py` | Accuracy paradox; P/R/F1; threshold tuning on val; ROC-AUC vs PR-AUC; class weights; calibration (ECE) |

### Part 2: deep learning (torch)
| # | Script | Core idea |
|---|--------|-----------|
| 08 | `08_autograd_backprop.py` | Scalar autograd engine (micrograd-style); numpy backprop + gradcheck; zero_grad/no_grad/detach; activation memory |
| 09 | `09_training_loop_debugging.py` | Init loss = ln C; overfit one batch; dropout / weight decay / early stopping with noisy labels; forgetting `eval()` |
| 10 | `10_init_vanishing_gradients.py` | Activation std through 20 layers by act × init; vanishing/exploding grads; residuals; clipping; symmetry |
| 11 | `11_normalization.py` | BN/LN/RMSNorm from scratch; batch dependence; post-norm vs pre-norm residual; small batches; LN scale pitfall |
| 12 | `12_optimizers_schedules.py` | LR range test; SGD/momentum/Nesterov/Adam; Adam+L2 vs AdamW; warmup+cosine; linear scaling rule |
| 13 | `13_cnn.py` | Conv by loops and im2col; param counts; receptive field; MLP vs CNN vs CNN+global pool on unseen positions |
| 14 | `14_rnn_lstm.py` | BPTT gradient decay; RNN vs LSTM vs GRU on long-range recall |
| 15 | `15_attention_transformer.py` | SDPA/MHA from scratch; why √d; permutation equivariance; positions; tiny transformer; KV cache |

Suggested order is numeric. 08 → 09 → 10 → 11 builds the "why deep nets
train" story, and 14 → 15 is the "why attention" story (15 solves the task 14 can't).

## Scoring

`run_all.py` parses every `[tag] key=value` line and checks **effects**, not
exact numbers: "leaky CV beats honest CV by > 0.2", "tiny batches hurt BN but
not LN", "the transformer recalls token 0 at length 50 only with positions".
To add a check, print a `kv(...)` line in the lesson and add a predicate to
`CHECKS`.

## Results that surprised me while building this (measured, `--quick`)

- **LayerNorm in a plain post-norm MLP is fragile** (11): with SGD it only
  trained at lr 1e-3, while BatchNorm trained from 1e-3 to 1e-1. Put the same
  norms in a **pre-norm residual** stack (the transformer layout) and all of
  them train at lr 0.1, and the no-norm net diverges there.
- **LayerNorm erases input scale** (11): LN(A·x) = LN(A·10x). The first
  version of the lesson put LN right after the input layer, and the spiral's
  radius (its main signal) disappeared. A non-linearity before the first norm
  fixed it.
- **Adam + L2 vs AdamW** (12): the measurable difference wasn't "big-gradient
  weights decay less" in aggregate. It was that weights with **no** gradient
  are wiped out (kept 4.5e-10 of their size) under L2-in-Adam, because the decay
  term gets normalised into a full-size step. AdamW shrank them uniformly
  (kept 0.55).
- **Gating extends memory, it doesn't remove the limit** (14): in 1500 steps
  LSTM/GRU solved recall at 20 steps, but nothing solved it at 30. A 2-layer
  transformer with positions solved length 50 (15).
- **The MLP barely learns translated glyphs at all** (13): 214k params, 0.38
  accuracy on the region it trained on, versus 0.998 for a 7k-param CNN with
  global max pooling (which also scored 0.999 on positions it never saw).
- **ROC-AUC vs PR-AUC** (07): the first version changed the dataset along with
  the class ratio, so ROC-AUC moved too. Holding the scorer fixed and changing
  only the ratio isolates the real effect: ROC-AUC stays at 0.92, PR-AUC falls
  from 0.92 to 0.17.
