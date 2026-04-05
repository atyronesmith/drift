#!/usr/bin/env python3
"""Project pre-digestion cost savings at various scales and API price points.

Uses corpus stats from predigest_corpus_stats.json to project savings.

Usage:
    uv run python scripts/cost_model.py
"""

import json
from pathlib import Path


def main():
    stats = json.loads(Path("results/predigest_corpus_stats.json").read_text())

    # Per-log averages from corpus
    n_logs = stats["n_logs"]
    avg_raw_tokens = stats["total_raw_tokens"] / n_logs
    avg_digest_tokens = stats["total_digest_tokens"] / n_logs
    savings_pct = stats["overall_token_savings_pct"]

    # Pipeline token estimates (from Experiment 14 analysis)
    # Summarize is 92% of pipeline; classify/route/solution add ~8%
    # Pre-digestion only affects the summarize step's input
    pipeline_overhead_tokens = 1578  # classify + route + solution input tokens
    output_tokens_per_log = 800  # all 4 steps combined output

    raw_pipeline = avg_raw_tokens + pipeline_overhead_tokens + output_tokens_per_log
    digest_pipeline = avg_digest_tokens + pipeline_overhead_tokens + output_tokens_per_log

    pipeline_savings_pct = (1 - digest_pipeline / raw_pipeline) * 100

    # API pricing (per 1M tokens)
    pricing = {
        "GPT-4o": {"input": 2.50, "output": 10.00},
        "GPT-4o-mini": {"input": 0.15, "output": 0.60},
        "Claude Sonnet": {"input": 3.00, "output": 15.00},
        "Claude Haiku": {"input": 0.25, "output": 1.25},
        "Llama 3.1 70B (hosted)": {"input": 0.35, "output": 0.40},
    }

    # Scale scenarios
    scales = [500, 1_000, 5_000, 10_000, 50_000, 100_000]

    print("ANSIBLE LOG ANALYSIS — PRE-DIGESTION COST MODEL")
    print("=" * 70)
    print()
    print(f"Corpus: {n_logs} production logs")
    print(f"Avg raw tokens/log:     {avg_raw_tokens:,.0f}")
    print(f"Avg digest tokens/log:  {avg_digest_tokens:,.0f}")
    print(f"Token savings:          {savings_pct}%")
    print(f"Pipeline overhead:      {pipeline_overhead_tokens:,} tokens (classify+route+solution)")
    print(f"Pipeline savings:       {pipeline_savings_pct:.1f}% (summarize dominates)")
    print()

    # Per-model cost tables
    for model, prices in pricing.items():
        input_price = prices["input"]
        output_price = prices["output"]

        print(f"--- {model} (${input_price}/M input, ${output_price}/M output) ---")
        print(f"{'Logs/month':>12s}  {'Raw cost':>10s}  {'Digest cost':>12s}  {'Saved':>10s}  {'Saved/yr':>10s}")
        print(f"{'':>12s}  {'':>10s}  {'':>12s}  {'':>10s}  {'':>10s}")

        for n in scales:
            raw_input_cost = n * avg_raw_tokens * input_price / 1_000_000
            raw_output_cost = n * output_tokens_per_log * output_price / 1_000_000
            raw_total = raw_input_cost + raw_output_cost

            digest_input_cost = n * avg_digest_tokens * input_price / 1_000_000
            digest_output_cost = n * output_tokens_per_log * output_price / 1_000_000
            digest_total = digest_input_cost + digest_output_cost

            saved = raw_total - digest_total
            saved_yr = saved * 12

            print(f"{n:>12,d}  ${raw_total:>9,.2f}  ${digest_total:>11,.2f}  ${saved:>9,.2f}  ${saved_yr:>9,.2f}")

        print()

    # Summary table: annual savings by model at 10K logs/month
    n = 10_000
    print("=" * 70)
    print(f"ANNUAL SAVINGS AT {n:,} LOGS/MONTH")
    print("=" * 70)
    print(f"{'Model':35s}  {'Monthly Raw':>12s}  {'Monthly Saved':>14s}  {'Annual Saved':>13s}")
    print("-" * 70)

    for model, prices in pricing.items():
        raw_cost = n * (avg_raw_tokens * prices["input"] + output_tokens_per_log * prices["output"]) / 1_000_000
        digest_cost = n * (avg_digest_tokens * prices["input"] + output_tokens_per_log * prices["output"]) / 1_000_000
        saved = raw_cost - digest_cost

        print(f"{model:35s}  ${raw_cost:>11,.2f}  ${saved:>13,.2f}  ${saved*12:>12,.2f}")

    print()
    print("Note: Output tokens are constant (pre-digestion only affects input).")
    print("Savings scale linearly with volume. No implementation cost modeled.")


if __name__ == "__main__":
    main()
