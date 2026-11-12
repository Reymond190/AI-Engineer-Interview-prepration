Each topic below has why it exists, an analogy, the core concepts, and 3 practice questions. Try the questions and send me your answers if you want them checked.

# Part 1: Classic algorithms

## 1. Linear and logistic regression
**Why it exists:** The simplest way to turn inputs into a prediction is a weighted sum. Linear regression predicts a number; logistic regression adapts it to yes/no questions.

**Analogy:** A restaurant bill. Each item (feature) has a price (weight), and the total is the prediction. Logistic regression then squashes that total into a 0–1 "chance of yes" dial.

**Core concepts**
- **Weight:** how much each feature pushes the prediction up or down.
- **Sigmoid:** an S-shaped function that turns any number into a probability between 0 and 1.
- **Cross-entropy:** a loss that punishes confident wrong answers very heavily.
- **Baseline:** the simple model you must beat before trying anything fancier.

**Practice**
1. Why can't plain linear regression be used directly to predict a probability?
2. A model says 99% "not spam" for an email that is spam. Why does cross-entropy punish this more than a 60% guess?
3. A weight on "age" is −0.8 in a loan-approval model. What does that tell you?

## 2. Decision tree
**Why it exists:** People wanted models that reason like humans, as a chain of simple questions.

**Analogy:** The game of 20 Questions. Each question should split the remaining possibilities as cleanly as possible.

**Core concepts**
- **Split:** a yes/no question on one feature (for example, "income > 50k?").
- **Gini/entropy:** measures of how mixed a group is; a good split makes groups purer.
- **Leaf:** the end of a path, where the prediction is made.
- **Overfitting:** a deep tree keeps asking questions until it memorises every training example.

**Practice**
1. A node has 50 cats and 50 dogs. Another has 99 cats and 1 dog. Which is purer, and which has higher entropy?
2. Why does an unlimited-depth tree get 100% training accuracy but do poorly on new data?
3. Name two ways to stop a tree from overfitting.

## 3. Random forest
**Why it exists:** Single trees are unstable; small data changes produce a very different tree. Averaging many different trees cancels out their individual quirks.

**Analogy:** Asking 500 doctors instead of one, where each doctor saw a different sample of patients and only looks at some of the symptoms. Their majority vote is more reliable.

**Core concepts**
- **Bagging:** train each tree on a random resample (with replacement) of the data.
- **Feature randomness:** each split considers only a random subset of features, so trees don't all look alike.
- **Variance reduction:** averaging decorrelated models smooths out noise.

**Practice**
1. Why would averaging 500 *identical* trees not help at all?
2. One feature is extremely predictive. How does feature randomness help here?
3. Does a random forest mainly fix bias or variance? Why?

## 4. Gradient boosting (XGBoost, LightGBM)
**Why it exists:** Instead of averaging independent models, build them in sequence so each one corrects the mistakes of the previous ones.

**Analogy:** A student doing practice exams. After each exam, they study only the questions they got wrong. Each round fixes the remaining errors.

**Core concepts**
- **Residual:** the gap between the current prediction and the truth.
- **Sequential trees:** each small tree is trained to predict the residuals.
- **Learning rate:** how much of each new tree's correction you add; smaller is slower but safer.
- **Bias reduction:** many weak models combine into a strong one.

**Practice**
1. Random forest trees can be trained in parallel; boosting trees can't. Why?
2. What happens if you add 10,000 boosting rounds with no early stopping?
3. You have a 50-column customer table and need to predict churn. Why is boosting usually the first serious model to try?

## 5. SVM
**Why it exists:** Many lines can separate two classes; SVM picks the one most likely to stay correct on new data, the one with the widest safety gap.

**Analogy:** Building the widest possible road between two villages. Only the houses on the road's edge determine where it goes.

**Core concepts**
- **Margin:** the gap between the boundary and the nearest points.
- **Support vectors:** the nearest points, which alone define the boundary.
- **Kernel trick:** computing as if the data were lifted into a higher dimension, where a straight boundary works, without actually doing the lift.

**Practice**
1. You delete a training point far from the boundary. Does the SVM boundary change?
2. Data forms a ring of one class around a blob of another. Why does a linear SVM fail, and what fixes it?
3. Why does a wider margin tend to generalise better?

## 6. k-Means
**Why it exists:** Sometimes there are no labels, and you just want to find natural groups.

**Analogy:** Placing k pizza shops in a city. Everyone goes to the nearest shop, then each shop moves to the centre of its customers. Repeat until nothing moves.

**Core concepts**
- **Centroid:** the centre of a cluster.
- **Assign/update loop:** assign points to the nearest centroid, then recompute centroids.
- **Elbow/silhouette:** tools for choosing k (where adding clusters stops helping, or how well-separated clusters are).
- **Sensitivity:** bad starting positions or unscaled features give bad clusters.

**Practice**
1. Features are "salary in rupees" and "age in years". Why will clustering ignore age unless you scale?
2. Two runs of k-means give different clusters. Why, and how do you reduce this?
3. Why does the loss always decrease as k grows, and why does that make "pick the lowest loss" a bad rule?

## 7. PCA
**Why it exists:** Many features are redundant. PCA compresses data into fewer dimensions while keeping as much information as possible.

**Analogy:** Photographing a 3D object. You pick the angle where the shadow shows the most detail, the direction where the object is most spread out.

**Core concepts**
- **Variance:** how spread out the data is along a direction; more spread means more information.
- **Principal component:** a direction of maximum variance; each next one is perpendicular to the earlier ones.
- **Projection:** dropping the low-variance directions.

**Practice**
1. Height in cm and height in inches are both features. How many components do you really need?
2. Why must you standardise features before PCA?
3. PCA is fit on labels-free data. Could the direction it discards be the one that matters for your prediction? Give an example.

# Part 2: Core concepts

## 8. Bias vs variance
**Analogy:** Archery. High bias means the arrows are tightly grouped but far from the bullseye (consistently wrong). High variance means they are scattered everywhere (unstable).

**Core concepts**
- **Bias:** error from a model too simple to capture the pattern (underfit).
- **Variance:** error from a model that reacts to noise in the training data (overfit).
- **Trade-off:** making a model more flexible usually lowers bias and raises variance.

**Practice**
1. Train error 2%, validation error 25%. Which problem is it?
2. Train error 30%, validation error 31%. Which problem, and what do you try?
3. Does adding more data fix high bias? Why or why not?

## 9. Overfitting fixes
**Analogy:** A student who memorised last year's answer key. The fixes are more varied practice (data, augmentation), limiting notes (regularisation, simpler model), studying with random teammates absent (dropout), and stopping before cramming sets in (early stopping).

**Core concepts**
- **Regularisation:** a penalty on complexity.
- **Dropout:** randomly switching off neurons in training so none becomes a crutch.
- **Early stopping:** stop when validation loss starts rising.
- **Augmentation:** create varied copies of data (flips, crops, noise).

**Practice**
1. Validation loss falls until epoch 12, then rises while train loss keeps falling. What do you do?
2. Why is dropout turned off at inference time?
3. You have 500 images. Which fixes would you try first?

## 10. L1 vs L2
**Analogy:** Packing for a trip with a baggage fee. L1 charges a flat rate per kilo, so you leave whole items at home. L2 charges for weight squared, so you keep everything but make each item lighter.

**Core concepts**
- **L1 (lasso):** penalty on the absolute values of the weights; pushes many to exactly zero (feature selection).
- **L2 (ridge):** penalty on squared weights; shrinks all of them, with no exact zeros.
- **Sparsity:** most weights being zero.

**Practice**
1. You have 10,000 features and suspect only 50 matter. L1 or L2?
2. Why does L2 punish one weight of 10 more than ten weights of 1?
3. Two features are highly correlated. How do L1 and L2 treat them differently?

## 11. Train/validation/test
**Analogy:** Textbook exercises (train), mock exams used to adjust your study strategy (validation), and the final exam (test). If you peek at the final repeatedly, your score stops meaning anything.

**Core concepts**
- **Hyperparameter:** a setting you choose rather than learn (tree depth, learning rate).
- **Validation set:** used to choose those settings.
- **Test set:** touched once, for an honest final number.

**Practice**
1. You tune 200 configurations against the test set and report the best. What's wrong?
2. Why is validation accuracy usually slightly optimistic?
3. Model A beats B on validation but loses on test. Which do you trust, and what do you do next?

## 12. Cross-validation
**Analogy:** Instead of one mock exam, every chapter takes a turn being the mock exam, and you average the scores.

**Core concepts**
- **k-fold:** split the data into k parts; train on k−1 and validate on the remaining one, rotating.
- **Stratified:** each fold keeps the same class proportions.
- **Cost:** k times the training work.

**Practice**
1. 1% of the samples are fraud. What can go wrong with non-stratified folds?
2. When is cross-validation more valuable, with 500 rows or 5 million?
3. Why is plain k-fold wrong for stock price data?

## 13. Gradient descent
**Analogy:** Walking down a foggy mountain by feeling the slope under your feet. Batch surveys the whole mountain before each step. SGD takes a step based on one pebble. Mini-batch checks a small patch.

**Core concepts**
- **Gradient:** the direction of steepest increase in loss; you step the opposite way.
- **Batch/SGD/mini-batch:** use all the data, one sample, or a small group per step.
- **Noise:** smaller batches give noisier steps, which are cheaper and can help escape bad spots.

**Practice**
1. Why is full-batch descent impractical with 100 million examples?
2. The loss curve with batch size 1 is very jagged. Why?
3. What is the practical reason mini-batches suit GPUs?

## 14. Optimizers
**Analogy:** A ball rolling downhill gains momentum and smooths out bumps. Adam also gives each parameter its own shoes, with big steps on flat ground and small steps on steep ground.

**Core concepts**
- **Momentum:** a running average of past gradients, which damps zig-zagging.
- **Adaptive learning rate:** per-parameter step sizes based on gradient history.
- **Weight decay:** shrinking weights each step.
- **AdamW:** applies the decay separately from the adaptive scaling, so the regularisation behaves as intended.

**Practice**
1. Loss zig-zags across a narrow valley. Which idea fixes this?
2. In plain Adam with an L2 penalty, why do parameters with large gradients get less regularisation?
3. You're fine-tuning a transformer. What optimizer do you choose by default, and why?

## 15. Learning rate
**Analogy:** Stride length going downhill. Giant leaps overshoot the valley; baby steps take forever. Warmup is walking carefully at first, and cosine decay is slowing down smoothly as you approach the bottom.

**Core concepts**
- **Divergence:** the loss explodes because steps are too large.
- **Warmup:** start with a tiny rate and ramp up, since early gradients are unreliable.
- **Cosine decay:** gradually reduce the rate for fine settling.

**Practice**
1. Loss becomes NaN at step 50. What is the first thing to try?
2. Loss is decreasing but painfully slowly. What might be wrong?
3. Why is warmup especially important with Adam on transformers?

## 16. Vanishing gradients
**Analogy:** The telephone game. A message whispered through 100 people fades to nothing, so the early people never learn what went wrong. Residual connections are a direct phone line that skips the chain.

**Core concepts**
- **Backpropagation:** the error signal is multiplied layer by layer going backward.
- **Vanishing:** many multiplications by numbers smaller than 1 shrink it to about zero.
- **ReLU/GELU:** activations that don't squash gradients the way sigmoid does.
- **Residual connection:** output = input + change, which gives gradients a shortcut.
- **Normalisation and good initialisation:** keep signal sizes stable.

**Practice**
1. Sigmoid's maximum slope is 0.25. What happens to a gradient after 20 such layers?
2. In a deep network, the first layers' weights barely change. What is the diagnosis?
3. Why do residual connections make 100-layer networks trainable?

## 17. BatchNorm vs LayerNorm
**Analogy:** BatchNorm grades on a curve, so your score depends on the classmates in the room. LayerNorm rescales each student's marks using only that student's own marks, so no classmates are needed.

**Core concepts**
- **Normalisation:** rescaling activations to a stable range.
- **BatchNorm:** statistics computed across the batch, per feature; works well for CNNs with large batches.
- **LayerNorm/RMSNorm:** statistics computed across the features of a single token; independent of batch size and sequence length.

**Practice**
1. Why does BatchNorm misbehave with batch size 2?
2. Why does BatchNorm behave differently in training and inference?
3. Sentences have varying lengths. Why does this favour LayerNorm in transformers?

## 18. Data leakage
**Analogy:** Seeing the answer key before the exam. The score looks brilliant and collapses in the real world.

**Core concepts**
- **Leakage:** training uses information that won't be available at prediction time.
- **Preprocessing leakage:** computing means or scalers on the full data before splitting.
- **Temporal leakage:** using the future to predict the past.
- **Rule:** split first, then fit all preprocessing on the training set only.

**Practice**
1. You standardise the whole dataset and then split. What leaked?
2. A model predicting hospital readmission uses the "discharge summary written after readmission" field. What is the problem?
3. Your model gets 99.8% on a hard problem. What are your first three checks?

Want me to put this into a doc you can keep, or grade your answers to any section?