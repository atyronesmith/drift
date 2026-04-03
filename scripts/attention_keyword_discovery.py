#!/usr/bin/env python3
"""Attention-based keyword rule discovery for classification fields.

Probes helpdesk tickets via MLX attention to discover which specific words
in the description the model attends to when classifying root_cause_category.
Computes TF-IDF-style discriminative keyword scores per category, then
validates by building a simple keyword classifier.

Usage:
    uv run python scripts/attention_keyword_discovery.py --tickets 50
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

# Stopwords to exclude from keyword candidates
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her",
    "my", "our", "your", "his", "its", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "when", "while", "as", "at", "by", "for",
    "in", "of", "on", "to", "with", "from", "not", "no", "so", "than",
    "too", "very", "just", "also", "how", "all", "each", "every", "any",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "then", "there", "here", "now", "about", "up", "out", "into", "over",
    "after", "before", "between", "under", "again", "further",
    # Chat template tokens that leak through subword merging
    "assistant", "system", "user", "endoftext",
    # Common IT ticket filler
    "hello", "please", "thanks", "thank", "help", "issue", "issues",
    "hey", "dear", "regards", "best", "team", "hi",
}


def format_ticket(t: dict) -> str:
    return (
        f"Ticket ID:    {t['id']}\n"
        f"Subject:      {t['subject']}\n"
        f"From:         {t['requesterEmail']}\n"
        f"Created:      {t['createdAt']}\n"
        f"Priority:     {t['priority']}\n"
        f"Category:     {t['category']}\n"
        f"\nDescription:\n{t['description']}"
    )


def load_tickets(path="datasets/helpdesk_tickets.jsonl", n=50):
    tickets = []
    with open(path) as f:
        for line in f:
            tickets.append(json.loads(line))
            if len(tickets) >= n:
                break
    return tickets


def find_description_token_range(input_tokens, sys_prompt_end):
    """Find token positions that belong to the Description field content."""
    text = ""
    token_offsets = []
    for i in range(sys_prompt_end, len(input_tokens)):
        start = len(text)
        text += input_tokens[i]
        end = len(text)
        token_offsets.append((start, end, i))

    # Find "Description:" label
    m = re.search(r"Description:\s*\n?", text)
    if not m:
        return {}, []

    desc_start = m.end()

    # Description tokens: everything after the label
    desc_positions = []
    for char_start, char_end, token_pos in token_offsets:
        if char_start >= desc_start:
            desc_positions.append(token_pos)

    return desc_positions


def merge_subword_tokens(tokens, positions):
    """Merge subword token pieces into whole words with their positions.

    Returns: [(word, [token_positions])]
    """
    # Filter out special/template tokens first
    SPECIAL_PATTERNS = {"<|", "|>", "<s>", "</s>", "[INST]", "[/INST]"}

    words = []
    current_word = ""
    current_positions = []

    for pos in positions:
        if pos >= len(tokens):
            continue
        tok = tokens[pos]

        # Skip special/template tokens
        if any(sp in tok for sp in SPECIAL_PATTERNS):
            if current_word.strip():
                words.append((current_word.strip().lower(), current_positions))
            current_word = ""
            current_positions = []
            continue

        # Detect word boundaries: token starts with a space or is punctuation
        is_boundary = (
            tok.startswith(" ") or
            tok.startswith("\n") or
            (not current_word) or
            (tok in {".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}"})
        )

        if is_boundary and current_word:
            words.append((current_word.strip().lower(), current_positions))
            current_word = tok
            current_positions = [pos]
        else:
            current_word += tok
            current_positions.append(pos)

    if current_word.strip():
        words.append((current_word.strip().lower(), current_positions))

    # Filter: only keep alphabetic words, skip stopwords and short tokens
    filtered = []
    for word, positions in words:
        clean = re.sub(r"[^a-z]", "", word)
        if len(clean) >= 3 and clean not in STOPWORDS:
            filtered.append((clean, positions))

    return filtered


def find_output_field_tokens(output_tokens, field_name):
    """Find the output token indices that produce a specific field's value.

    Looks for "field_name": "value" in the JSON output token stream.
    Returns only the tokens that make up the value itself (between quotes).
    """
    text = ""
    tokens_text = []  # (index, token_str, cumulative_text_end)

    for i, probe in enumerate(output_tokens):
        token = probe["token"] if isinstance(probe, dict) else probe.token
        text += token
        tokens_text.append((i, token, len(text)))

    # Find the field key pattern and extract the value
    pattern = f'"{field_name}":\\s*"([^"]*)"'
    m = re.search(pattern, text)
    if not m:
        return [], None

    value = m.group(1)
    val_start = m.start(1)
    val_end = m.end(1)

    # Find which token indices correspond to the value span
    result_indices = []
    char_pos = 0
    for i, probe in enumerate(output_tokens):
        token = probe["token"] if isinstance(probe, dict) else probe.token
        tok_start = char_pos
        tok_end = char_pos + len(token)
        # Token overlaps with value span
        if tok_end > val_start and tok_start < val_end:
            result_indices.append(i)
        char_pos = tok_end

    return result_indices, value


def extract_keyword_attention(result, sys_prompt_end):
    """Extract per-word attention from root_cause_category output tokens to description words.

    Returns: [(word, total_attention_weight)]
    """
    # Find description token positions in input
    desc_positions = find_description_token_range(result.input_tokens, sys_prompt_end)
    if not desc_positions:
        return [], None

    desc_positions_set = set(desc_positions)

    # Merge subword tokens into words
    desc_words = merge_subword_tokens(result.input_tokens, desc_positions)

    # Build reverse map: token_position -> (word_index, word)
    pos_to_word = {}
    for word_idx, (word, positions) in enumerate(desc_words):
        for pos in positions:
            pos_to_word[pos] = (word_idx, word)

    # Find output tokens for root_cause_category
    rc_tokens, rc_value = find_output_field_tokens(result.output_tokens, "root_cause_category")
    if not rc_tokens or not rc_value:
        return [], None

    # Aggregate attention from root_cause output tokens to description words
    word_weights = defaultdict(float)
    for idx in rc_tokens:
        if idx >= len(result.output_tokens):
            continue
        probe = result.output_tokens[idx]
        top_attended = probe["top_attended"] if isinstance(probe, dict) else probe.top_attended
        for pos, weight in top_attended:
            if pos in pos_to_word and not math.isnan(weight):
                _, word = pos_to_word[pos]
                word_weights[word] += weight

    word_attention = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
    return word_attention, rc_value


def compute_keyword_scores(category_keywords):
    """Compute TF-IDF-style discriminative scores per category.

    Args:
        category_keywords: {category: [{word: attention_weight}]} per ticket

    Returns:
        {category: [(word, score, within_count, other_count)]}
    """
    categories = list(category_keywords.keys())
    n_categories = len(categories)

    # Aggregate per-word stats within each category
    # For each category: mean attention per word, count of tickets containing word
    category_word_stats = {}
    for cat in categories:
        ticket_words = category_keywords[cat]  # list of {word: weight} dicts
        n_tickets = len(ticket_words)
        if n_tickets == 0:
            continue

        # Aggregate: total weight and ticket count per word
        word_total = defaultdict(float)
        word_ticket_count = defaultdict(int)
        for tw in ticket_words:
            seen = set()
            for word, weight in tw.items():
                word_total[word] += weight
                if word not in seen:
                    word_ticket_count[word] += 1
                    seen.add(word)

        # Mean attention per word
        word_mean = {w: word_total[w] / n_tickets for w in word_total}
        category_word_stats[cat] = {
            "word_mean": word_mean,
            "word_ticket_count": word_ticket_count,
            "n_tickets": n_tickets,
        }

    # Compute IDF: for each word, how many categories have it?
    word_category_count = defaultdict(int)
    for cat, stats in category_word_stats.items():
        for word in stats["word_mean"]:
            word_category_count[word] += 1

    # Score = mean_attention * log(n_categories / categories_with_word)
    # Also require word appears in >= 50% of tickets in the category
    results = {}
    for cat, stats in category_word_stats.items():
        scored = []
        n_tickets = stats["n_tickets"]
        for word, mean_attn in stats["word_mean"].items():
            ticket_count = stats["word_ticket_count"][word]
            cat_count = word_category_count[word]

            # Frequency within category (must appear in >= 40% of tickets)
            within_freq = ticket_count / n_tickets
            if within_freq < 0.4:
                continue

            # IDF component
            idf = math.log(n_categories / cat_count) if cat_count < n_categories else 0.1

            # Count appearances in other categories' tickets
            other_count = 0
            for other_cat, other_stats in category_word_stats.items():
                if other_cat != cat:
                    other_count += other_stats["word_ticket_count"].get(word, 0)

            score = mean_attn * idf
            scored.append((word, score, ticket_count, other_count))

        scored.sort(key=lambda x: x[1], reverse=True)
        results[cat] = scored[:15]  # top 15 per category

    return results


def validate_keywords(keyword_scores, all_tickets_data):
    """Build a simple keyword classifier and test accuracy.

    Args:
        keyword_scores: {category: [(word, score, ...)]}
        all_tickets_data: [(ticket, ground_truth_category)]

    Returns:
        (accuracy, predictions, confusion)
    """
    # Build weighted keyword sets per category (top 10)
    category_keywords = {}
    for cat, scored in keyword_scores.items():
        category_keywords[cat] = {word: score for word, score, _, _ in scored[:10]}

    correct = 0
    total = 0
    predictions = []
    confusion = defaultdict(lambda: defaultdict(int))

    for ticket, true_cat in all_tickets_data:
        desc = ticket["description"].lower()

        # Score each category by keyword matches
        cat_scores = {}
        for cat, keywords in category_keywords.items():
            cat_scores[cat] = sum(
                score for word, score in keywords.items() if word in desc
            )

        if cat_scores:
            predicted = max(cat_scores, key=cat_scores.get)
            if cat_scores[predicted] == 0:
                predicted = None
        else:
            predicted = None

        if predicted == true_cat:
            correct += 1
        total += 1

        predictions.append({
            "ticket_id": ticket["id"][:8],
            "true": true_cat,
            "predicted": predicted,
            "correct": predicted == true_cat,
            "scores": {k: round(v, 3) for k, v in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:3]},
        })

        confusion[true_cat][predicted or "none"] += 1

    accuracy = correct / total if total else 0
    return accuracy, predictions, dict(confusion)


def main():
    parser = argparse.ArgumentParser(description="Attention-based keyword discovery")
    parser.add_argument("--tickets", type=int, default=50, help="Number of tickets")
    parser.add_argument("--save", default="results/keyword_discovery", help="Output directory")
    args = parser.parse_args()

    from drift.providers.mlx_remote import MLXRemoteProvider
    provider = MLXRemoteProvider(model=MODEL)

    system_prompt = open("tasks/helpdesk_triage/triage_system.md").read()
    tickets = load_tickets(n=args.tickets)

    print("=" * 70)
    print("ATTENTION-BASED KEYWORD DISCOVERY")
    print("=" * 70)
    print(f"Model:   {MODEL}")
    print(f"Tickets: {len(tickets)}")
    print()

    # ── Phase 1: Estimate system prompt boundary ──
    print("Estimating system prompt boundary...", end=" ", flush=True)
    boundary_result = provider.probe_attention(
        system=system_prompt, prompt="test", max_tokens=1, top_k_attend=5,
    )
    sys_prompt_end = 0
    for i, tok in enumerate(boundary_result.input_tokens):
        if "test" in tok.lower():
            sys_prompt_end = i
            break
    if sys_prompt_end == 0:
        sys_prompt_end = max(boundary_result.n_input - 10, 0)
    print(f"token {sys_prompt_end}")
    print()

    # ── Phase 2 & 3: Probe tickets and extract keyword attention ──
    category_keywords = defaultdict(list)  # {category: [ticket_word_weights]}
    all_tickets_data = []  # [(ticket, category)]
    skipped = 0

    for i, ticket in enumerate(tickets):
        input_text = format_ticket(ticket)
        print(f"  {i+1:3d}/{len(tickets)} {ticket['id'][:8]}... ", end="", flush=True)

        t0 = time.time()
        result = provider.probe_attention(
            system=system_prompt,
            prompt=input_text,
            max_tokens=800,
            top_k_attend=100,
        )
        elapsed = time.time() - t0

        word_attention, rc_value = extract_keyword_attention(result, sys_prompt_end)

        if not rc_value or not word_attention:
            print(f"{elapsed:.1f}s  SKIP (no root_cause found)")
            skipped += 1
            continue

        # Store per-ticket word weights for this category
        word_dict = dict(word_attention)
        category_keywords[rc_value].append(word_dict)
        all_tickets_data.append((ticket, rc_value))

        top3 = ", ".join(f"{w}={a:.3f}" for w, a in word_attention[:3])
        print(f"{elapsed:.1f}s  rc={rc_value:<20s} top: {top3}")

    print()
    print(f"Probed {len(all_tickets_data)} tickets, skipped {skipped}")
    print(f"Categories found: {sorted(category_keywords.keys())}")
    print(f"Tickets per category: {', '.join(f'{k}={len(v)}' for k, v in sorted(category_keywords.items()))}")
    print()

    # ── Phase 4: TF-IDF keyword scoring ──
    print("=" * 70)
    print("DISCRIMINATIVE KEYWORD SCORES (TF-IDF style)")
    print("=" * 70)
    print()

    keyword_scores = compute_keyword_scores(category_keywords)

    for cat in sorted(keyword_scores.keys()):
        scored = keyword_scores[cat]
        n = len(category_keywords[cat])
        print(f"  {cat} ({n} tickets):")
        if not scored:
            print(f"    (no discriminative keywords found)")
        for word, score, within, other in scored[:10]:
            print(f"    {word:<25s} score={score:.3f}  "
                  f"(in {within}/{n} tickets, {other} in other categories)")
        print()

    # ── Phase 5: Validation ──
    print("=" * 70)
    print("KEYWORD CLASSIFIER VALIDATION")
    print("=" * 70)
    print()

    accuracy, predictions, confusion = validate_keywords(keyword_scores, all_tickets_data)

    print(f"  Accuracy: {accuracy:.1%} ({sum(1 for p in predictions if p['correct'])}/{len(predictions)})")
    print()

    # Show confusion matrix
    all_cats = sorted(set(list(confusion.keys()) + [
        p for row in confusion.values() for p in row.keys()
    ]))
    header = "True \\ Predicted"
    print(f"  {header:<25s}", end="")
    for cat in all_cats:
        print(f" {cat[:12]:>12s}", end="")
    print()
    print(f"  {'─'*25}", end="")
    for _ in all_cats:
        print(f" {'─'*12}", end="")
    print()

    for true_cat in all_cats:
        if true_cat not in confusion:
            continue
        print(f"  {true_cat:<25s}", end="")
        for pred_cat in all_cats:
            count = confusion[true_cat].get(pred_cat, 0)
            if count > 0:
                print(f" {count:>12d}", end="")
            else:
                print(f" {'·':>12s}", end="")
        print()

    print()

    # Show misclassifications
    wrong = [p for p in predictions if not p["correct"]]
    if wrong:
        print(f"  Misclassifications ({len(wrong)}):")
        for p in wrong:
            print(f"    {p['ticket_id']} true={p['true']:<20s} pred={p['predicted']}")
    print()

    # ── Save results ──
    os.makedirs(args.save, exist_ok=True)

    # Save keyword rules in YAML-ready format
    keyword_rules = {}
    for cat, scored in keyword_scores.items():
        keyword_rules[cat] = [word for word, _, _, _ in scored[:10]]

    output = {
        "experiment": "attention_keyword_discovery",
        "model": MODEL,
        "n_tickets": len(all_tickets_data),
        "n_skipped": skipped,
        "categories": {cat: len(tix) for cat, tix in category_keywords.items()},
        "accuracy": round(accuracy, 4),
        "keyword_rules": keyword_rules,
        "keyword_scores": {
            cat: [{"word": w, "score": round(s, 4), "within": wc, "other": oc}
                  for w, s, wc, oc in scored[:10]]
            for cat, scored in keyword_scores.items()
        },
        "confusion": confusion,
        "predictions": predictions,
    }

    out_path = os.path.join(args.save, "keyword_discovery.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    # Print YAML-ready keyword rule
    print()
    print("=" * 70)
    print("PROPOSED KEYWORD RULE (YAML)")
    print("=" * 70)
    print()
    print("- path: assessment.root_cause_category")
    print("  type: keyword")
    print("  field: description")
    print("  keywords:")
    for cat in sorted(keyword_rules.keys()):
        words = keyword_rules[cat]
        print(f"    {cat}: {json.dumps(words)}")
    print(f"  default: user_error")
    print()


if __name__ == "__main__":
    main()
