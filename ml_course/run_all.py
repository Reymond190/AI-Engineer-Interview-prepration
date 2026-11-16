"""
run_all.py - run the ML/DL fundamentals lessons and check their OUTCOMES.

Same idea as langgraph_course/ and finetune_course/run_all.py, but simpler to
extend: every lesson prints `[tag] key=value ...` lines (common.kv), which are
parsed into {tag: {key: value}}; each check is a predicate over that dict. So a
check reads like the claim it verifies, e.g. "leaky CV beats honest CV by 0.2":
    lambda r: r["leak-select"]["outside_cv"] > r["leak-select"]["inside_pipeline"] + 0.2
Checks test the EFFECT a lesson teaches, not exact numbers, so they survive
seeds and hardware changes. A missing tag (lesson crashed early) fails the check.

Run in WSL, env `ft` (vLLM can stay up: these models are tiny):
    python ml_course/run_all.py                    # all lessons, --quick
    python ml_course/run_all.py 09 11              # selected lessons
    python ml_course/run_all.py --full --save results_ml_full.json
    python ml_course/run_all.py --compare a.json b.json
Full output of each lesson: ml_course/outputs/logs/<lesson>.log
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "outputs" / "logs"

R = dict[str, dict[str, object]]
Check = tuple[str, Callable[[R], bool]]

CHECKS: dict[str, list[Check]] = {
    "00": [("selecting on test is optimistic", lambda r: r["selection"]["best_on_test"] > r["selection"]["fresh_holdout"] + 0.05),
           ("feature selection outside CV leaks", lambda r: r["leak-select"]["outside_cv"] > r["leak-select"]["inside_pipeline"] + 0.2),
           ("GroupKFold removes group leak", lambda r: r["leak-group"]["kfold"] > r["leak-group"]["group_kfold"] + 0.2),
           ("shuffled CV on time series is optimistic", lambda r: r["leak-time"]["shuffled_r2"] > r["leak-time"]["walk_forward_r2"] + 0.3),
           ("k-fold less noisy than one split", lambda r: r["cv-variance"]["kfold_std"] < r["cv-variance"]["single_std"])],
    "01": [("normal equation == sklearn", lambda r: r["linreg"]["normal_vs_sklearn"] < 1e-6),
           ("collinearity makes coefficients unstable", lambda r: r["collinear"]["coef_std_max"] > 100 * r["collinear"]["coef_std_min"]),
           ("CE gradient >> MSE gradient when confidently wrong",
            lambda r: abs(r["ce-vs-mse"]["ce_grad_wrong"]) > 100 * abs(r["ce-vs-mse"]["mse_grad_wrong"])),
           ("logistic GD works", lambda r: r["logreg"]["ours_acc"] > 0.9),
           ("analytic gradient passes gradcheck", lambda r: r["gradcheck"]["rel_err"] < 1e-6),
           ("naive softmax overflows, stable doesn't", lambda r: r["softmax"]["naive_has_nan"] is True and abs(r["softmax"]["stable_sum"] - 1) < 1e-6),
           ("softmax regression on digits", lambda r: r["softmax-digits"]["test_acc"] > 0.9)],
    "02": [("lr above 2/L diverges", lambda r: r["lr-threshold"]["above_2_over_L"] > 10 * r["lr-threshold"]["below_2_over_L"]),
           ("momentum beats GD at high kappa", lambda r: r["conditioning"]["momentum_k1000"] < r["conditioning"]["gd_k1000"]),
           ("standardising fixes conditioning", lambda r: r["scaling"]["scaled_kappa"] < 10 < r["scaling"]["raw_kappa"]),
           ("Adam beats GD on Rosenbrock", lambda r: r["rosenbrock"]["adam"] < r["rosenbrock"]["gd"]),
           ("lr decay / bigger batch lower the SGD floor",
            lambda r: r["sgd-floor"]["bs1_decay"] < r["sgd-floor"]["bs1_const"] and r["sgd-floor"]["bs32_const"] < r["sgd-floor"]["bs1_const"])],
    "03": [("test error is U-shaped in complexity", lambda r: r["complexity"]["test_d15"] > r["complexity"]["test_best"]),
           ("low degree = bias, high degree = variance",
            lambda r: r["bias-variance"]["bias_d1"] > r["bias-variance"]["bias_d3"] and r["bias-variance"]["var_d12"] > r["bias-variance"]["var_d3"]),
           ("ridge rescues the flexible model", lambda r: r["ridge"]["best"] < r["ridge"]["alpha0"] / 2),
           ("more data cures variance, not bias", lambda r: r["learning-curve"]["d12_test_n1000"] < r["learning-curve"]["d1_test_n1000"]),
           ("lasso zeroes weights, ridge doesn't", lambda r: r["l1-l2"]["ridge_zeros"] == 0 and r["l1-l2"]["lasso_zeros"] >= 30)],
    "04": [("purer node has lower gini", lambda r: r["impurity"]["gini_99_1"] < r["impurity"]["gini_50_50"]),
           ("deep tree memorises", lambda r: r["tree"]["deep_train"] == 1.0 and r["tree"]["deep_test"] < 1.0),
           ("forest beats a single tree", lambda r: r["forest"]["rf"] > r["forest"]["single"] + 0.03),
           ("our GBM ~ sklearn GBM", lambda r: abs(r["boosting"]["ours_lr01"] - r["boosting"]["sklearn"]) < 0.25),
           ("impurity importance credits a random ID", lambda r: r["importance"]["id_impurity"] > 5 * abs(r["importance"]["id_permutation"]))],
    "05": [("Pegasos SVM works", lambda r: r["linear-svm"]["ours_acc"] > 0.9),
           ("small C -> more support vectors", lambda r: r["svm-C"]["sv_small_C"] > r["svm-C"]["sv_large_C"]),
           ("kernel == explicit feature dot product", lambda r: r["kernel-identity"]["abs_diff"] < 1e-9),
           ("RBF solves circles, linear can't", lambda r: r["kernels"]["rbf"] > r["kernels"]["linear"] + 0.2),
           ("huge gamma overfits", lambda r: r["gamma"]["train_g1000"] > r["gamma"]["test_g1000"] + 0.2),
           ("scaling fixes kNN", lambda r: r["knn-scaling"]["scaled"] > r["knn-scaling"]["raw"] + 0.1),
           ("distances concentrate in high d", lambda r: r["curse"]["ratio_d1000"] > r["curse"]["ratio_d2"] + 0.5),
           ("noise dims break kNN", lambda r: r["curse-knn"]["acc_0"] > r["curse-knn"]["acc_200"] + 0.1)],
    "06": [("our k-means == sklearn", lambda r: abs(r["kmeans"]["ours_inertia"] / r["kmeans"]["sklearn_inertia"] - 1) < 0.01),
           ("k-means++ beats random init", lambda r: r["init"]["pp_median"] < r["init"]["random_median"]),
           ("silhouette finds k", lambda r: r["choose-k"]["silhouette_best_k"] == r["choose-k"]["true_k"]),
           ("DBSCAN beats k-means on moons", lambda r: r["nonconvex"]["dbscan_ari"] > r["nonconvex"]["kmeans_ari"] + 0.3),
           ("SVD PCA == sklearn", lambda r: r["pca"]["max_component_diff"] < 1e-8),
           ("unscaled PCA dominated by one feature", lambda r: r["pca-scaling"]["raw_pc1_ratio"] > 0.9)],
    "07": [("accuracy paradox", lambda r: r["paradox"]["always_neg_acc"] > 0.95 and r["paradox"]["always_neg_recall"] == 0),
           ("hand P/R/F1 == sklearn", lambda r: r["prf-check"]["match"] is True),
           ("tuned threshold beats 0.5", lambda r: r["threshold"]["f1_tuned"] > r["threshold"]["f1_at_05"]),
           ("ROC-AUC stable, PR-AUC collapses", lambda r: r["auc"]["roc_rare"] > 0.85 and r["auc"]["pr_rare"] < r["auc"]["pr_balanced"] - 0.4),
           ("class weights raise recall", lambda r: r["class-weight"]["recall_balanced"] > r["class-weight"]["recall_none"] + 0.3),
           ("NB miscalibrated, isotonic fixes it",
            lambda r: r["calibration"]["ece_nb"] > 2 * r["calibration"]["ece_lr"] and r["calibration"]["ece_nb_iso"] < r["calibration"]["ece_nb"] / 2)],
    "08": [("scalar autograd == torch", lambda r: r["autograd"]["max_grad_diff"] < 1e-9),
           ("micrograd MLP learns", lambda r: r["micrograd-mlp"]["train_acc"] > 0.65),
           ("numpy backprop passes gradcheck", lambda r: r["gradcheck"]["worst_rel_err"] < 1e-5),
           ("numpy backprop == torch", lambda r: r["vs-torch"]["max_diff"] < 1e-9),
           ("grads accumulate without zero_grad", lambda r: r["torch-gotchas"]["accumulated_x"] == 3.0)],
    "09": [("init loss ~ ln(C)", lambda r: abs(r["init-loss"]["default_init"] / r["init-loss"]["expected"] - 1) < 0.1),
           ("overconfident init has high loss", lambda r: r["init-loss"]["last_layer_std10"] > 2 * r["init-loss"]["expected"]),
           ("can overfit one batch", lambda r: r["overfit-batch"]["final_loss"] < 0.01),
           ("dropout lowers val loss vs none", lambda r: r["regularise"]["dropout_val_loss"] < r["regularise"]["none_val_loss"]),
           ("forgetting eval() hurts", lambda r: r["eval-mode"]["acc_eval"] > r["eval-mode"]["acc_train_mode_mean"]),
           ("final model works", lambda r: r["final"]["test_acc"] > 0.95)],
    "10": [("He keeps ReLU activations alive, Xavier doesn't",
            lambda r: r["act-std"]["relu_he_L20"] > 100 * r["act-std"]["relu_xavier_L20"]),
           ("sigmoid gradients vanish", lambda r: r["grad-flow"]["sigmoid_ratio"] < 1e-6),
           ("deep sigmoid fails, residual trains",
            lambda r: r["deep-train"]["sigmoid_acc"] < 0.5 and r["deep-train"]["residual_acc"] > 0.9),
           ("clipping bounds the norm", lambda r: abs(r["clip"]["grad_norm_after"] - 1) < 1e-3 and r["clip"]["grad_norm_before"] > 1e3),
           ("constant init keeps units identical", lambda r: r["symmetry"]["distinct_hidden_units"] == 1)],
    "11": [("hand BN/LN/RMSNorm == torch", lambda r: max(v for v in r["parity"].values()) < 1e-4),
           ("BN depends on batch-mates, LN doesn't",
            lambda r: r["batch-dependence"]["batchnorm"] > 0.1 and r["batch-dependence"]["layernorm"] < 1e-5),
           ("BN refuses batch 1 in train mode", lambda r: r["batch1"]["bn_train_ok"] is False and r["batch1"]["bn_eval_ok"] is True),
           ("post-norm: none fails, BN trains at lr 0.1",
            lambda r: r["postnorm"]["none_best"] < 0.5 and r["postnorm"]["bn_lr01"] > 0.9),
           ("pre-norm residual: no norm diverges at lr 0.1, LN/RMS fine",
            lambda r: r["prenorm"]["none_lr01"] == "diverged" and min(r["prenorm"]["ln_lr01"], r["prenorm"]["rms_lr01"]) > 0.9),
           ("tiny batches hurt BN only", lambda r: r["small-batch"]["ln_bs2"] > r["small-batch"]["bn_bs2"] + 0.2),
           ("LN erases input scale", lambda r: r["ln-scale-invariance"]["max_diff_r_vs_10r"] < 0.01)],
    "12": [("range test finds a blow-up point", lambda r: r["lr-range"]["blowup_lr"] > r["lr-range"]["lr_at_min_loss"]),
           ("momentum/Adam beat plain SGD", lambda r: min(r["optimizers"]["momentum_acc"], r["optimizers"]["adam_acc"]) > r["optimizers"]["sgd_acc"]),
           ("too-high Adam lr hurts", lambda r: r["optimizers"]["adam_too_high_acc"] < r["optimizers"]["adam_acc"] - 0.05),
           ("L2-in-Adam wipes gradient-free weights; AdamW shrinks uniformly",
            lambda r: r["adamw"]["l2_dead_kept"] < 1e-3 and r["adamw"]["adamw1_dead_kept"] > 0.3),
           ("decay beats constant at high lr", lambda r: r["schedule"]["cosine_acc"] > r["schedule"]["const_acc"]),
           ("linear lr scaling helps big batches", lambda r: r["batch-scaling"]["bs256_scaled"] < r["batch-scaling"]["bs256_fixed"])],
    "13": [("loops/im2col == F.conv2d", lambda r: max(r["conv-parity"]["loops_vs_torch"], r["conv-parity"]["im2col_vs_torch"]) < 1e-9),
           ("output size formula", lambda r: r["conv-parity"]["out_hw"] == r["conv-parity"]["formula"]),
           ("global-pool CNN generalises to unseen positions",
            lambda r: r["translation"]["cnn_pool_shifted"] > 0.9 and max(r["translation"]["mlp_shifted"], r["translation"]["cnn_flat_shifted"]) < 0.5)],
    "14": [("LSTM carries more gradient than RNN", lambda r: r["bptt-grad"]["lstm_len50"] > 100 * r["bptt-grad"]["rnn_len50"]),
           ("gated cells remember 20 steps, RNN doesn't",
            lambda r: min(r["memory"]["lstm"], r["memory"]["gru"]) > 0.9 and r["memory"]["rnn"] < 0.3)],
    "15": [("hand attention == SDPA", lambda r: r["sdpa-parity"]["max_diff"] < 1e-4),
           ("unscaled softmax saturates", lambda r: r["sqrt-d"]["unscaled_maxp_1024"] > 0.9 and r["sqrt-d"]["scaled_maxp_1024"] < 0.3),
           ("hand MHA == nn.MultiheadAttention", lambda r: r["mha-parity"]["max_diff"] < 1e-4),
           ("attention is permutation-equivariant", lambda r: r["perm-equivariance"]["max_diff"] < 1e-4),
           ("transformer solves length-50 recall only with positions",
            lambda r: r["first-token"]["with_pos"] > 0.9 and r["first-token"]["no_pos"] < 0.5),
           ("KV cache == full recompute", lambda r: r["kv-cache"]["max_diff"] < 1e-4)],
}

KV_LINE = re.compile(r"^\[([\w-]+)\] (.*)$", re.M)


def parse_value(s: str) -> object:
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse(out: str) -> R:
    results: R = {}
    for tag, rest in KV_LINE.findall(out):
        results[tag] = {k: parse_value(v) for k, v in (p.split("=", 1) for p in rest.split() if "=" in p)}
    return results


def lessons(selected: list[str]) -> list[Path]:
    files = sorted(HERE.glob("[0-9][0-9]_*.py"))
    return [p for p in files if any(p.name.startswith(s + "_") for s in selected)] if selected else files


def run_lesson(path: Path, quick: bool, timeout: int) -> dict:
    key = path.name.split("_")[0]
    t0 = time.perf_counter()
    cmd = [sys.executable, str(path)] + (["--quick"] if quick else [])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL, cwd=HERE)
        out, code = proc.stdout + "\n" + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, code = ((e.stdout or "") if isinstance(e.stdout, str) else "") + "\nTIMEOUT", -1
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{path.stem}.log").write_text(out, encoding="utf-8")
    parsed = parse(out)
    checks = {}
    for desc, pred in CHECKS.get(key, []):
        try:
            checks[desc] = bool(pred(parsed))
        except (KeyError, TypeError):   # tag missing (crash) or wrong type (e.g. 'diverged' vs float)
            checks[desc] = False
    return {"lesson": path.stem, "exit": code, "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
            "results": parsed, "tail": out[-800:] if code else ""}


def print_table(runs: list[dict]) -> None:
    print(f"\n{'lesson':<36} {'exit':>5} {'checks':>7} {'seconds':>8}")
    tp = tt = 0
    for r in runs:
        p, t = sum(r["checks"].values()), len(r["checks"])
        tp, tt = tp + p, tt + t
        print(f"{r['lesson']:<36} {r['exit']:>5} {p:>3}/{t:<3} {r['seconds']:>8.0f}")
    print(f"{'TOTAL':<36} {'':>5} {tp:>3}/{tt:<3} {sum(r['seconds'] for r in runs):>8.0f}")


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    bl = {r["lesson"]: r for r in b["runs"]}
    for r in a["runs"]:
        o = bl.get(r["lesson"])
        cell = lambda x: f"{sum(x['checks'].values())}/{len(x['checks'])} {x['seconds']:5.0f}s" if x else "-"  # noqa: E731
        print(f"{r['lesson']:<36} {cell(r):>14} {cell(o):>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", help="lesson prefixes, e.g. 09 11 (default: all)")
    ap.add_argument("--full", action="store_true", help="full-length runs instead of --quick")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--save", help="write results JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    runs = []
    for path in lessons(args.lessons):
        r = run_lesson(path, quick=not args.full, timeout=args.timeout)
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"{r['lesson']:<36} exit={r['exit']} {r['seconds']:>6.0f}s  failed={failed or '-'}", flush=True)
        if r["exit"]:
            print("  " + r["tail"].replace("\n", "\n  "))
        runs.append(r)
    print_table(runs)
    if args.save:
        Path(args.save).write_text(json.dumps({"mode": "full" if args.full else "quick", "runs": runs}, indent=2))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
