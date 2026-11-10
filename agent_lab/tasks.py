"""
tasks.py - benchmark tasks. Math tasks have known answers (auto-graded);
research/forex tasks are graded by reading report.md (and the verifier).

The math tasks are Project-Euler-style problems, restated. A 9B model may have
memorised famous answers, which is why the harness insists on code + verification;
`pe_custom_*` tasks are variants with changed parameters that can't be memorised.
"""

TASKS: dict[str, dict] = {
    # --- math / algorithms (known answers) -------------------------------------------
    "pe_primes": {
        "prompt": "Find the sum of all prime numbers below 2,000,000.",
        "answer": "142913828922",
    },
    "pe_collatz": {
        "prompt": "For the Collatz map (n -> n/2 if even, 3n+1 if odd), which starting number under one million "
                  "produces the longest chain before reaching 1?",
        "answer": "837799",
    },
    "pe_semiprimes": {
        "prompt": "How many composite integers n < 10^8 have exactly two (not necessarily distinct) prime factors?",
        "answer": "17427258",
    },
    "pe_hamming": {
        "prompt": "A generalised Hamming number of type 100 has no prime factor larger than 100. "
                  "How many such numbers are there that do not exceed 10^9?",
        "answer": "2944730",
    },
    "pe_divisor_squares": {
        "prompt": "Let sigma2(n) be the sum of the squares of the divisors of n. Find the sum of all n with "
                  "0 < n < 64,000,000 such that sigma2(n) is a perfect square.",
        "answer": "1922364685",
    },
    "pe_custom_consecutive_divisors": {
        "prompt": "Count the integers 1 < n < 5,000,000 for which n and n+1 have the same number of positive divisors.",
        "answer": None,  # fill in after a verified run; parameters changed from the famous version
    },
    # --- research / forex (open-ended) ---------------------------------------------
    "fx_seasonality": {
        "prompt": "Using EURUSD H1 candles for the last 3 years (fetch_fx), test whether hour-of-day return "
                  "seasonality exists and survives out of sample. Use walk-forward validation, include spread costs, "
                  "report effect sizes with confidence intervals, and say clearly whether any edge is tradable.",
        "answer": None,
    },
    "fx_next_day": {
        "prompt": "Build a model that outputs the probability that EURUSD closes higher tomorrow than today (D1 data, "
                  "at least 10 years). Evaluate it walk-forward with Brier score and log loss against a 50% baseline "
                  "and a simple momentum baseline, then give the probability for the next trading day with honest caveats.",
        "answer": None,
    },
    "fx_regimes": {
        "prompt": "Characterise volatility regimes in GBPUSD and USDJPY H4 data over the last 5 years (e.g. GARCH or "
                  "realised-volatility clustering). Identify the current regime for each pair and how persistent "
                  "regimes have historically been. Save charts.",
        "answer": None,
    },
}
