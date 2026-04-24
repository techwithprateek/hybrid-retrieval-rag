"""
Synthetic Knowledge Base Generator
Generates new support ticket entries for the knowledge base using an LLM via litellm.

Why litellm?
- litellm is a unified Python wrapper around 100+ LLM providers (OpenAI, Anthropic,
  Ollama, Gemini, Together AI, …).  The same `litellm.completion()` call works with
  any of them — you just change the model string.
- This means you can generate KB data using a free local model (Ollama) for
  development/testing, then switch to a cloud model for higher quality production data.

litellm supports 100+ providers so you can swap the model with a single flag:
  - OpenAI:      --model gpt-4o-mini          (default)
  - Anthropic:   --model claude-3-haiku-20240307
  - Ollama:      --model ollama/mistral
  - Together AI: --model together_ai/mistral-7b-instruct-v0.1
  - Gemini:      --model gemini/gemini-1.5-flash
  - Any other:   --model <litellm-model-string>

Usage
-----
# Generate 10 new entries (appended to the existing knowledge base):
    python generate_kb.py --count 10

# Generate entries for specific categories only:
    python generate_kb.py --categories Payment "Order" --count 5

# Use a different model (Ollama local):
    python generate_kb.py --model ollama/mistral --count 5 --api-base http://localhost:11434

# Write output to a separate file instead of appending:
    python generate_kb.py --count 10 --output data/synthetic_kb.csv

# Dry-run — print generated entries without saving:
    python generate_kb.py --count 3 --dry-run
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

# Load environment variables from .env (if present) so the API key is available
# without the user having to export it manually in every terminal session.
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default LLM used when no --model flag is provided.
# gpt-4o-mini is cheap and fast — good for bulk KB generation.
DEFAULT_MODEL = "gpt-4o-mini"

# Default path for the knowledge base CSV, relative to this script file.
# Using __file__ means this works regardless of where you run the script from.
DEFAULT_KB_PATH = Path(__file__).parent / "data" / "knowledge_base.csv"

# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------
# This dictionary defines every supported category, its possible subcategories,
# and which journey stages apply to it.
#
# Journey stages:
# - Pre-Purchase  : issue happens before the customer places an order
# - Purchase      : issue happens during checkout / payment
# - Post-Purchase : issue happens after order confirmation (delivery, refunds, etc.)
#
# This config is used to:
# 1. Pick generation targets (what category/subcategory to ask the LLM to write)
# 2. Detect duplicates — if a (category, subcategory) pair already exists in
#    the KB, we skip it so the generated data stays novel and diverse.
CATEGORY_CONFIG: dict[str, dict] = {
    "Payment": {
        "subcategories": [
            "Failed Transaction",
            "UPI Failure",
            "Refund Not Received",
            "OTP Not Received",
            "Duplicate Charge",
            "Net Banking Error",
            "EMI Issue",
            "International Payment Declined",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Account": {
        "subcategories": [
            "Login Failure",
            "Account Locked",
            "Profile Update Issue",
            "KYC Verification Failed",
            "Account Deactivated",
            "Password Reset Issue",
            "Email Verification Failed",
            "Two-Factor Auth Issue",
        ],
        # Account issues can occur at any stage of the customer journey
        "stages": ["Pre-Purchase", "Purchase", "Post-Purchase"],
    },
    "Order": {
        "subcategories": [
            "Order Not Placed",
            "Wrong Item Delivered",
            "Order Delayed",
            "Order Cancellation Request",
            "Return Request",
            "Missing Item in Package",
            "Order Marked Delivered but Not Received",
            "Exchange Request",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Delivery": {
        "subcategories": [
            "Delivery Agent Issue",
            "Package Damaged",
            "Delivery Address Change",
            "Failed Delivery Attempt",
            "Delivery to Wrong Address",
            "Customs / Import Duty Issue",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Technical": {
        "subcategories": [
            "App Crash",
            "Slow Loading",
            "Search Not Working",
            "Payment Gateway Error",
            "Checkout Error",
            "Image Not Loading",
            "Notification Not Working",
            "Dark Mode Issue",
        ],
        # Technical issues typically surface before or during checkout
        "stages": ["Pre-Purchase", "Purchase"],
    },
    "Coupon": {
        "subcategories": [
            "Coupon Not Applied",
            "Cashback Not Credited",
            "Referral Bonus Missing",
            "Discount Mismatch",
            "Coupon Expired Prematurely",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Subscription": {
        "subcategories": [
            "Subscription Not Activated",
            "Subscription Auto-Renewal Dispute",
            "Subscription Downgrade Issue",
            "Subscription Benefits Not Unlocked",
            "Cancellation Not Processed",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Wallet": {
        "subcategories": [
            "Wallet Balance Not Updated",
            "Wallet Payment Failed",
            "Wallet Top-Up Failure",
            "Wallet Withdrawal Issue",
            "Expired Wallet Credits",
        ],
        "stages": ["Purchase", "Post-Purchase"],
    },
    "Fraud": {
        "subcategories": [
            "Unauthorized Transaction",
            "Account Hacked",
            "Phishing Complaint",
            "Seller Fraud",
            "Fake Product Received",
        ],
        # Fraud is almost always reported after the incident (post-purchase)
        "stages": ["Post-Purchase"],
    },
}

# Column names that every row in the knowledge base CSV must have.
# issue_id is assigned by this script; the LLM generates the other five fields.
KB_COLUMNS = [
    "issue_id",
    "category",
    "subcategory",
    "journey_stage",
    "issue_description",
    "resolution_steps",
]

# ---------------------------------------------------------------------------
# LLM system prompt for KB generation
# ---------------------------------------------------------------------------
# This prompt is sent as the "system" message so the LLM understands its role
# (KB curator) and the exact JSON schema it must produce.
#
# Why a strict JSON schema?
# - We programmatically parse the response and write it to a CSV.
# - Any deviation (extra fields, wrong types, markdown wrapping) would break parsing.
# - We also instruct the LLM NOT to include issue_id because we assign that ourselves
#   to guarantee uniqueness and sequential ordering.
GENERATION_SYSTEM_PROMPT = """\
You are a customer support knowledge base curator. Your job is to generate realistic,
detailed support ticket templates for an e-commerce platform.

You MUST respond with a JSON array containing exactly the number of entries requested.
Each entry must follow this schema:
{
  "category": "<string>",
  "subcategory": "<string>",
  "journey_stage": "<Pre-Purchase | Purchase | Post-Purchase>",
  "issue_description": "<2-4 sentence description of the issue and its common causes>",
  "resolution_steps": "<numbered steps as a single string, e.g. '1. Step one. 2. Step two.'>"
}

Rules:
- issue_description must describe the customer-facing problem and common root causes.
- resolution_steps must be practical, numbered, and 4-8 steps long.
- Do NOT include issue_id — it will be assigned automatically.
- Keep language clear and customer-friendly.
- Do NOT repeat entries from the existing knowledge base.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_existing_kb(path: Path) -> list[dict]:
    """
    Load all rows from an existing knowledge base CSV and return them as a
    list of dicts (one dict per row).  Returns an empty list if the file
    does not exist yet (first run scenario).

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dicts, each mapping column name → value string.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _next_issue_id(existing: list[dict]) -> int:
    """
    Find the highest issue_id currently in the KB and return the next integer.
    This ensures new entries always get a unique, incrementing ID.

    Args:
        existing: List of row dicts loaded from the KB CSV.

    Returns:
        The next available issue_id (1 if the KB is empty).
    """
    if not existing:
        return 1
    # Parse all numeric issue_ids and take the max; ignore any non-numeric values
    ids = [int(row["issue_id"]) for row in existing if row.get("issue_id", "").isdigit()]
    return max(ids, default=0) + 1


def _build_existing_summary(existing: list[dict]) -> str:
    """
    Build a compact bullet-point summary of every entry already in the KB.

    This summary is injected into the LLM prompt so the model knows which
    (category, subcategory, journey_stage) combinations already exist and
    should NOT be repeated in the new batch.

    Args:
        existing: List of row dicts from the KB.

    Returns:
        A newline-delimited string of "- Category / Subcategory (Stage)" lines,
        or "None yet." if the KB is empty.
    """
    if not existing:
        return "None yet."
    lines = [
        f"- {r['category']} / {r['subcategory']} ({r['journey_stage']})"
        for r in existing
    ]
    return "\n".join(lines)


def _pick_targets(
    categories: list[str] | None,
    count: int,
    existing: list[dict],
) -> list[dict[str, str]]:
    """
    Select (category, subcategory, journey_stage) combinations for the LLM
    to write new KB entries for.

    Strategy:
    1. Build the set of (category, subcategory) pairs that already exist in the KB.
    2. For every requested category, iterate over its subcategories and add any
       pair that is NOT already in the KB to the candidate pool.
    3. If the pool is smaller than `count`, all unique unseen combos are returned.
    4. If the pool is empty (everything already exists), fall back to allowing
       duplicate combos — the LLM is instructed to vary the content.
    5. Shuffle the pool for variety, then take the first `count` items.

    Args:
        categories: Optional list of category names to restrict generation to.
                    If None, all categories in CATEGORY_CONFIG are used.
        count: How many target combos to return.
        existing: Current KB rows (used to detect already-covered combos).

    Returns:
        A list of dicts, each with keys: category, subcategory, journey_stage.
    """
    # Build a set of (category, subcategory) pairs already in the KB for O(1) lookup
    existing_pairs = {
        (r["category"], r["subcategory"]) for r in existing
    }

    pool: list[dict[str, str]] = []
    # Use provided categories or default to all configured categories
    chosen_cats = categories or list(CATEGORY_CONFIG.keys())

    for cat in chosen_cats:
        cfg = CATEGORY_CONFIG.get(cat)
        if cfg is None:
            # Warn if the user passed a category name that isn't in our config
            print(f"  Warning: unknown category '{cat}', skipping.", file=sys.stderr)
            continue
        for sub in cfg["subcategories"]:
            if (cat, sub) not in existing_pairs:
                # Pick a random valid journey stage for this category
                stage = random.choice(cfg["stages"])
                pool.append({"category": cat, "subcategory": sub, "journey_stage": stage})

    if not pool:
        # All (category, subcategory) combos already exist in the KB.
        # Fall back: allow potential content-level duplicates (the LLM is
        # prompted to vary issue descriptions and resolution steps).
        for cat in chosen_cats:
            cfg = CATEGORY_CONFIG.get(cat, {})
            for sub in cfg.get("subcategories", []):
                stage = random.choice(cfg.get("stages", ["Purchase"]))
                pool.append({"category": cat, "subcategory": sub, "journey_stage": stage})

    # Randomise order so we don't always generate the same categories first
    random.shuffle(pool)
    # Return at most `count` targets
    return pool[:count]


def _generate_batch(
    targets: list[dict[str, str]],
    existing: list[dict],
    model: str,
    api_base: str | None,
    temperature: float,
    max_retries: int,
) -> list[dict]:
    """
    Ask the LLM to generate knowledge base entries for the given targets
    in a single API call, retrying with exponential backoff on failure.

    Why batch calls?
    - Sending multiple entries per call reduces per-call overhead (latency,
      token overhead for the system prompt) and is more cost-efficient.
    - The --batch-size flag controls how many entries are in each call.

    Why exponential backoff?
    - LLM APIs can return rate-limit errors (429) or transient network errors.
    - Exponential backoff (2s, 4s, 8s, …) avoids hammering the API while
      still recovering automatically from short-lived outages.

    Args:
        targets: List of (category, subcategory, journey_stage) dicts for
                 the LLM to write entries for.
        existing: All KB rows seen so far (used to build the "avoid duplicates"
                  summary in the prompt).
        model: litellm model string (e.g. "gpt-4o-mini", "ollama/mistral").
        api_base: Optional base URL override for local models like Ollama.
        temperature: LLM sampling temperature (higher = more creative/varied).
        max_retries: Max number of attempts before giving up on this batch.

    Returns:
        A list of raw entry dicts parsed from the LLM's JSON response.

    Raises:
        RuntimeError: If all retry attempts are exhausted without a valid response.
    """
    # Build the "existing entries" summary to prevent the LLM from repeating content
    existing_summary = _build_existing_summary(existing)

    # Build a numbered list of target combos for the user message
    target_spec = "\n".join(
        f"  {i+1}. category={t['category']}, subcategory={t['subcategory']}, "
        f"journey_stage={t['journey_stage']}"
        for i, t in enumerate(targets)
    )

    # Compose the user message: tell the LLM exactly what to generate and
    # what already exists (to avoid duplicates)
    user_message = (
        f"Generate exactly {len(targets)} knowledge base entries for the following "
        f"category/subcategory/journey_stage combinations:\n{target_spec}\n\n"
        f"Existing entries to avoid duplicating:\n{existing_summary}\n\n"
        "Return a JSON array of objects matching the schema described in the system prompt."
    )

    # Build the litellm call kwargs.
    # response_format={"type": "json_object"} is supported by most OpenAI-compatible
    # models and forces the output to be a valid JSON string.
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    # api_base is only needed for local models (e.g. Ollama at localhost:11434)
    if api_base:
        kwargs["api_base"] = api_base

    # Retry loop with exponential backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(**kwargs)
            raw = response.choices[0].message.content.strip()

            # Parse the raw JSON string from the LLM
            data = json.loads(raw)

            # Some models wrap the array in a top-level JSON object instead of
            # returning the array directly.  e.g. {"entries": [...]} instead of [...].
            # We try several common wrapper key names before giving up.
            if isinstance(data, dict):
                for key in ("entries", "data", "items", "results", "knowledge_base"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    # No known wrapper key found — take the first list value in the dict
                    lists = [v for v in data.values() if isinstance(v, list)]
                    data = lists[0] if lists else []

            return data

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            # The LLM returned something we couldn't parse as JSON — retry
            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s, …
                print(
                    f"  Attempt {attempt} failed ({exc}). Retrying in {wait}s…",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"LLM returned unparseable response after {max_retries} attempts."
                ) from exc

        except Exception as exc:
            # litellm wraps provider-specific errors (rate limits, auth errors, …)
            # under generic exceptions — retry with backoff for transient issues
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  API error on attempt {attempt}: {exc}. Retrying in {wait}s…",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise


def _validate_entry(entry: dict) -> bool:
    """
    Check that a generated entry dict contains all five required fields with
    non-empty string values.  Invalid entries are skipped rather than crashing
    the whole generation run.

    Args:
        entry: A raw dict from the LLM's JSON response.

    Returns:
        True if the entry is valid and safe to write to the CSV; False otherwise.
    """
    required = {"category", "subcategory", "journey_stage", "issue_description", "resolution_steps"}
    # all() short-circuits on the first falsy value, so this is efficient
    return all(entry.get(k, "").strip() for k in required)


def _write_csv(path: Path, rows: list[dict], mode: str = "a") -> None:
    """
    Write (or append) a list of row dicts to a CSV file.

    Args:
        path: Destination CSV file path.
        rows: List of dicts with keys matching KB_COLUMNS.
        mode: "a" to append to an existing file (default), "w" to overwrite.
              The header row is written automatically when creating a new file
              or when mode="w".
    """
    # Only write the header when creating a new file or overwriting an existing one
    write_header = mode == "w" or not path.exists() or path.stat().st_size == 0
    with path.open(mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KB_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Define and parse all command-line flags.

    Returns:
        An argparse.Namespace with all flag values accessible as attributes.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic knowledge base entries using an LLM via litellm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # show the module docstring as CLI help epilog
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of new entries to generate (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "litellm model string (default: gpt-4o-mini). "
            "Examples: claude-3-haiku-20240307, ollama/mistral, "
            "together_ai/mistral-7b-instruct-v0.1, gemini/gemini-1.5-flash"
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Optional API base URL (e.g. http://localhost:11434 for Ollama).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",         # accepts one or more space-separated values
        default=None,
        metavar="CATEGORY",
        help=(
            "Limit generation to specific categories. "
            f"Available: {', '.join(CATEGORY_CONFIG)}. "
            "Defaults to all categories."
        ),
    )
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=DEFAULT_KB_PATH,
        help=f"Path to the knowledge base CSV (default: {DEFAULT_KB_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to --kb-path (appends to existing file). "
            "Pass a different path to write to a separate file."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of entries to request per LLM call (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "LLM sampling temperature (default: 0.8). "
            "Higher values produce more varied content; lower values are more consistent."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM call on failure (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated entries to stdout without saving to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible target selection.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full KB generation pipeline:
    1. Parse CLI flags.
    2. Load the existing KB to know what already exists.
    3. Pick generation targets (unseen category/subcategory combos).
    4. Split targets into batches and call the LLM for each batch.
    5. Validate every generated entry.
    6. Assign sequential issue IDs and write to CSV (or print for dry run).
    """
    args = parse_args()

    # Set the random seed for reproducibility if provided.
    # This ensures _pick_targets() selects the same targets every time when
    # using the same seed, which is useful for testing or CI.
    if args.seed is not None:
        random.seed(args.seed)

    # Determine the output path:
    # - If --output is given, write to that separate file.
    # - Otherwise, append directly to the knowledge base CSV.
    output_path: Path = args.output or args.kb_path

    # Print a summary of the run configuration for transparency
    print(f"Model       : {args.model}")
    print(f"Count       : {args.count}")
    print(f"Batch size  : {args.batch_size}")
    print(f"KB path     : {args.kb_path}")
    print(f"Output path : {'[dry run]' if args.dry_run else output_path}")
    if args.categories:
        print(f"Categories  : {', '.join(args.categories)}")
    print()

    # Load existing KB entries — used for deduplication and issue_id assignment
    existing = _load_existing_kb(args.kb_path)
    print(f"Existing entries: {len(existing)}")

    # Pick the (category, subcategory, journey_stage) combos to generate
    targets = _pick_targets(args.categories, args.count, existing)
    if not targets:
        print("No new targets to generate — all category/subcategory combos already exist.")
        return

    actual_count = len(targets)
    if actual_count < args.count:
        # Inform the user that we ran out of unique combos before reaching --count
        print(
            f"Note: only {actual_count} unique unseen combinations available "
            f"(requested {args.count})."
        )

    # Determine the starting issue_id for new entries
    next_id = _next_issue_id(existing)

    # Accumulator for all successfully validated generated entries
    generated: list[dict] = []

    # Split targets into batches of --batch-size for efficient LLM calls
    batches = [targets[i : i + args.batch_size] for i in range(0, len(targets), args.batch_size)]

    print(f"Generating {actual_count} entries in {len(batches)} batch(es)…\n")

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Batch {batch_num}/{len(batches)} ({len(batch)} entries)…", end=" ", flush=True)
        try:
            raw_entries = _generate_batch(
                targets=batch,
                # Pass both existing KB rows AND already-generated rows to the
                # deduplication prompt so the LLM doesn't repeat within a run
                existing=existing + generated,
                model=args.model,
                api_base=args.api_base,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
        except RuntimeError as exc:
            # Log the failure and continue with the next batch rather than aborting
            print(f"\nFailed: {exc}")
            continue

        # Validate each entry and assign sequential issue IDs
        for entry in raw_entries:
            if not _validate_entry(entry):
                # Skip entries that are missing required fields
                print(f"\n  Skipped invalid entry: {entry}", file=sys.stderr)
                continue
            row = {
                "issue_id": next_id,
                "category": entry.get("category", "").strip(),
                "subcategory": entry.get("subcategory", "").strip(),
                "journey_stage": entry.get("journey_stage", "").strip(),
                "issue_description": entry.get("issue_description", "").strip(),
                "resolution_steps": entry.get("resolution_steps", "").strip(),
            }
            generated.append(row)
            next_id += 1  # increment so the next entry gets a unique ID

        print(f"OK (+{len(raw_entries)} entries)")

    if not generated:
        print("\nNo valid entries were generated.")
        return

    print(f"\nGenerated {len(generated)} valid entries.")

    if args.dry_run:
        # Print results to stdout without touching any file on disk
        print("\n--- DRY RUN OUTPUT ---")
        writer = csv.DictWriter(sys.stdout, fieldnames=KB_COLUMNS)
        writer.writeheader()
        writer.writerows(generated)
    else:
        # Ensure the output directory exists (e.g. data/ folder)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use "w" (overwrite) when writing to a separate output file so it starts
        # clean; use "a" (append) when writing back to the original KB file.
        mode = "w" if output_path != args.kb_path else "a"
        _write_csv(output_path, generated, mode=mode)
        print(f"Saved to: {output_path}")

        # Report the new total entry count for the user's information
        total = len(existing) + len(generated) if mode == "a" else len(generated)
        print(f"Total entries in KB: {total}")


# Run main() only when this script is executed directly (not when imported)
if __name__ == "__main__":
    main()
