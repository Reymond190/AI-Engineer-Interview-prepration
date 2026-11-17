# ML & DL fundamentals: the simple version

A plain-language companion to [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md). Each
answer is the short idea first, then the one number worth remembering. When
an interviewer pushes for detail, switch to the full guide. Numbers in
parentheses are lesson numbers; "measured" numbers come from `--quick` runs.

---

## 1. Data and evaluation

**Why train / validation / test?**

- **Train:** the model learns from it.
- **Validation:** *you* learn from it (pick settings, model, threshold).
- **Test:** look **once**, at the end.

If you peek at the test set to choose things, your score is fake. Measured
(00): on random labels, the "best" of 50 models looked like 0.625 and was
really 0.525.

**k-fold CV?**
Split into k parts and validate on each part once. Much less noisy than one
split. Use stratified folds for classes, group folds when the same person
appears twice, and time-ordered folds for time series.

**Data leakage?**
The model secretly gets information it won't have in real life. Measured
(00):

- Picking features using all the data first: 0.88 on pure noise (honest: 0.47).
- Same patient in train and test: 0.995 (honest: 0.53).
- Shuffling time-series data: R² 0.995 (honest: 0.11).

**Fix:** put every "fit" step (scaler, selector, encoder) in a Pipeline so it
only learns from the training fold.

**Which metric?**
Pick based on which mistake costs more. Accuracy lies when a class is rare:
"always say no" gets 98% when only 2% are positive.

- **Precision:** of the ones I flagged, how many were right?
- **Recall:** of the real positives, how many did I catch?
- **F1:** a balance of the two.
- **ROC-AUC:** how well positives are ranked above negatives. It ignores how
  rare positives are.
- **PR-AUC:** shows the pain when positives are rare. Measured (07): at 0.5%
  positives, ROC-AUC still says 0.92 but PR-AUC says 0.17.

**Imbalanced data: what do you do?**
In order:

1. Use the right metric.
2. Move the decision threshold (07: F1 0.16 → 0.33).
3. Class weights.
4. Resample, but only inside the training fold.
5. Get more positives.

**Calibration?**
When the model says 80%, is it right 80% of the time? Naive Bayes is badly
calibrated when features repeat (it counts the same evidence 5 times).
Isotonic or Platt scaling fixes it without changing the ranking.

---

## 2. Linear models and optimisation

**Linear regression?**
Find the line that minimises squared error. There's a formula, but use a
stable solver (QR/SVD). If two features are near-copies, the individual
weights go crazy (±33,000) while predictions stay fine. Ridge fixes it.

**Why cross-entropy instead of MSE for classification?**
When the model is *confidently wrong*, MSE's gradient almost vanishes, so it
barely learns from its worst mistakes. Cross-entropy's gradient stays strong
(measured: −1.0 vs −0.00009).

**Softmax overflow?**
Subtract the biggest number before `exp`. Same answer, no NaN.

**Learning rate on a simple bowl-shaped loss?**
Too big and it explodes (the limit is lr < 2/curvature). A long, thin valley
(high "condition number") makes plain gradient descent slow. Momentum helps
a lot (6160 → 987 steps).

**Why scale features?**
Mixed units (metres and millimetres) make the loss a long thin valley, so
gradient descent crawls. Measured (02): never converged unscaled, 78 steps
scaled. Trees don't care about scaling; most other models do.

**Optimizers in one line:**

- **Momentum:** a rolling ball. It keeps going in a consistent direction.
- **RMSProp:** each weight gets its own step size.
- **Adam:** both, plus a start-up correction. The default choice.

**SGD (mini-batches) vs full batch?**
Mini-batch gradients are noisy. With a fixed lr you bounce around the bottom
forever, so decay the lr or use bigger batches. The noise also helps you
find flatter, better minima.

---

## 3. Bias, variance, regularisation

**Bias vs variance?**

- **Bias:** too simple, wrong in the same way every time (underfit).
- **Variance:** too flexible, changes a lot with each dataset (overfit).

Picture a dartboard: bias = consistently off-centre; variance = scattered.

**How do you tell which one you have?**

- **Train bad, val bad:** bias. Use a bigger model or better features.
- **Train good, val bad:** variance. Get more data, regularise, simplify.
- More data fixes variance, not bias.

**L1 vs L2?**

- **L2 (ridge):** shrinks every weight a bit. Good with correlated features.
- **L1 (lasso):** pushes useless weights to exactly 0, so it does feature
  selection. Measured (03): kept all 5 real features out of 50 and dropped
  the other 45.
- Scale features first. Tune the strength on validation.

---

## 4. Trees and ensembles

**Decision tree?**
Ask yes/no questions about features. At each step pick the question that
makes the groups most "pure". With no depth limit it memorises (train 1.0,
test 0.92).

**Bagging and random forest?**

- **Bagging:** train many trees on random resamples and average them.
- **Random forest:** same, but each split only sees some features, so the
  trees disagree more, and averaging helps more.
- Measured (04): one tree 0.81, bagging 0.92, forest 0.94. Adding more
  trees never overfits.

**Gradient boosting?**
Add small trees one at a time, each fixing the previous trees' mistakes.
Reduces bias. *Can* overfit, so use a small learning rate and early
stopping. XGBoost/LightGBM are still the go-to for tables of data.

**Feature importance trap?**
Built-in tree importance favours columns with many unique values, even a
random ID. Use permutation importance on held-out data, or SHAP.

---

## 5. SVM, kNN, high dimensions

**SVM?**
Draw the boundary with the widest possible gap between classes. Only the
points near the gap ("support vectors") matter. C controls the trade-off:
small C = wide gap that allows mistakes; large C = strict.

**Kernel trick?**
Get the effect of mapping data into a richer space without actually doing
it, by only computing similarities. Circles: linear 0.62, RBF kernel 1.00.
Slow on lots of data.

**kNN pitfalls?**

- Unscaled features ruin it (0.88 → 0.56 with one big-unit noise feature).
- **Curse of dimensionality:** in high dimensions everything is about the
  same distance away, so "nearest" means little.
- k=1 memorises.

---

## 6. Unsupervised

**k-means?**
Repeat: assign each point to its nearest centre, then move each centre to
the middle of its points. It can get stuck in a bad answer, so use
k-means++ and several restarts. It only finds round blobs (it fails on
moon shapes, where DBSCAN works).

**Choosing k?** Elbow plot or best silhouette score.

**PCA?**
Find the directions where the data spreads most and keep the top few.
Standardise first, or one big-unit feature takes over (proline was 99.8% of
PC1). High variance ≠ useful for prediction. t-SNE/UMAP are only for
pictures.

---

## 7. Neural net training

**Backprop?**
The chain rule, run backwards from the loss. Each layer passes the gradient
back, multiplied by its own local slope. Backward costs about 2× forward.

**What uses memory?** Weights, gradients, optimizer state, and the saved
activations from the forward pass (these grow with batch size).

**PyTorch gotchas:**

- Gradients **add up** unless you zero them.
- `model.eval()` turns off dropout and fixes BatchNorm. Forget it and your
  results are noisy.

**Net won't train? Debug checklist:**

1. Starting loss ≈ ln(number of classes)? (3 classes → ~1.10.)
2. Can it overfit one small batch to ~0? If not, there's a bug.
3. Scale up and watch train and val together.
4. Add regularisation once it overfits.
5. NaN? Lower lr, clip gradients, check the inputs.

**Regularising nets?** Weight decay, dropout, data augmentation, early
stopping (keep the best checkpoint), smaller model, label smoothing.
Restoring the best checkpoint gave every setup its best score (09).

---

## 8. Making deep nets trainable

**Vanishing / exploding gradients?**
Each layer multiplies the gradient by a number. Many layers = that number
to a big power, so it shrinks to 0 or blows up. Sigmoid is bad (its slope is
at most 0.25). Fixes: ReLU, good init, normalisation, residual connections,
gradient clipping.

**Xavier vs He init?**
Both keep the signal the same size through layers. Xavier for tanh, He for
ReLU (ReLU kills half the signal, so He doubles the variance). Setting all
weights equal = all neurons stay identical forever.

**Residual connections?**
y = x + F(x). The gradient always has a straight "highway" back. Measured
(10): 100 layers without residuals = random guessing; with residuals = 1.0.

**BatchNorm vs LayerNorm vs RMSNorm?**

- **BatchNorm:** normalise across the batch. Breaks with tiny batches and
  behaves differently in training and eval.
- **LayerNorm:** normalise within each example. Same in training and eval.
  Used in transformers.
- **RMSNorm:** a simpler LayerNorm (Llama, Qwen).

**Pre-LN vs post-LN?** Normalising *before* each block (pre-LN) trains more
stably.

---

## 9. Optimizers in practice

**Picking a learning rate?** LR range test: raise the lr until the loss
blows up, then use about 1/10 of that.

**SGD vs Adam?** Adam is faster and needs less tuning. Tuned SGD with
momentum can match it on CNNs.

**Adam + L2 vs AdamW?** With Adam + L2, weight decay gets mixed into Adam's
scaling, so it's uneven across weights. AdamW applies decay separately and
evenly. Use AdamW. Don't decay biases or norm layers.

**Warmup + decay?** Warmup: start gently while Adam's estimates are
unreliable. Decay: calm down at the end to settle into the minimum.
Measured (12): high lr, constant 0.90 vs warmup + cosine 0.97.

**Batch size?** Double the batch → roughly double the lr (SGD), or ×√2 for
Adam. This stops working past a point.

---

## 10. Architectures

**CNN?**
Small filters slide across the image and reuse the same weights everywhere,
so there are far fewer parameters (1.8k vs 201M for a dense layer). Global
pooling makes it find a pattern anywhere: trained on the left half of
images, it still got 0.999 on the right half. An MLP got 0.26.

**RNN vs LSTM vs GRU?**
RNNs forget early tokens because gradients fade step by step. LSTM/GRU add
gates and an additive memory lane, so they remember longer, but still not
forever (all failed at length 30 here).

**Attention?**
Each token asks "who's relevant to me?" (Q against K), then takes a
weighted mix of their values (V).

- **Divide by √d:** otherwise the scores get huge, softmax picks one token,
  and gradients die.
- **Causal mask:** a token can't look at the future.
- **Multi-head:** several smaller attentions looking for different things.
- **Positions:** attention ignores word order, so you must add it (RoPE
  etc.). Without positions: 0.19; with: 1.00.

**Transformer cost and KV cache?**
Attention costs grow with length². FlashAttention does the same math without
storing the big table. The **KV cache** remembers past keys and values so
each new token is cheap. Size: ~128 KB per token for an 8B model, so ~4 GB
for one 32k-token conversation.
