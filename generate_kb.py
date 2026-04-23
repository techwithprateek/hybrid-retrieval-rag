"""
Synthetic Knowledge Base Generator
Generates new support ticket entries for the knowledge base using an LLM via litellm.

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

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_KB_PATH = Path(__file__).parent / "data" / "knowledge_base.csv"

# All supported categories and their typical subcategories / journey stages
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
        "stages": ["Post-Purchase"],
    },
}

KB_COLUMNS = [
    "issue_id",
    "category",
    "subcategory",
    "journey_stage",
    "issue_description",
    "resolution_steps",
]

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
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_kb(path: Path) -> list[dict]:
    """Load existing knowledge base entries from a CSV file."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _next_issue_id(existing: list[dict]) -> int:
    """Return the next available issue_id."""
    if not existing:
        return 1
    ids = [int(row["issue_id"]) for row in existing if row.get("issue_id", "").isdigit()]
    return max(ids, default=0) + 1


def _build_existing_summary(existing: list[dict]) -> str:
    """Build a compact summary of existing entries to avoid duplicates."""
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
    Pick (category, subcategory, journey_stage) combinations to generate.
    Prefer under-represented categories and avoid combinations already present.
    """
    existing_pairs = {
        (r["category"], r["subcategory"]) for r in existing
    }

    pool: list[dict[str, str]] = []
    chosen_cats = categories or list(CATEGORY_CONFIG.keys())

    for cat in chosen_cats:
        cfg = CATEGORY_CONFIG.get(cat)
        if cfg is None:
            print(f"  Warning: unknown category '{cat}', skipping.", file=sys.stderr)
            continue
        for sub in cfg["subcategories"]:
            if (cat, sub) not in existing_pairs:
                stage = random.choice(cfg["stages"])
                pool.append({"category": cat, "subcategory": sub, "journey_stage": stage})

    if not pool:
        # Fall back to any combination (may duplicate, LLM is instructed to vary)
        for cat in chosen_cats:
            cfg = CATEGORY_CONFIG.get(cat, {})
            for sub in cfg.get("subcategories", []):
                stage = random.choice(cfg.get("stages", ["Purchase"]))
                pool.append({"category": cat, "subcategory": sub, "journey_stage": stage})

    random.shuffle(pool)
    return pool[:count]


def _generate_batch(
    targets: list[dict[str, str]],
    existing: list[dict],
    model: str,
    api_base: str | None,
    temperature: float,
    max_retries: int,
) -> list[dict]:
    """Call the LLM to generate a batch of knowledge base entries."""
    existing_summary = _build_existing_summary(existing)
    target_spec = "\n".join(
        f"  {i+1}. category={t['category']}, subcategory={t['subcategory']}, "
        f"journey_stage={t['journey_stage']}"
        for i, t in enumerate(targets)
    )

    user_message = (
        f"Generate exactly {len(targets)} knowledge base entries for the following "
        f"category/subcategory/journey_stage combinations:\n{target_spec}\n\n"
        f"Existing entries to avoid duplicating:\n{existing_summary}\n\n"
        "Return a JSON array of objects matching the schema described in the system prompt."
    )

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if api_base:
        kwargs["api_base"] = api_base

    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(**kwargs)
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            # Some models wrap the array in a key
            if isinstance(data, dict):
                # Try common wrapper keys
                for key in ("entries", "data", "items", "results", "knowledge_base"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    # Take the first list value found
                    lists = [v for v in data.values() if isinstance(v, list)]
                    data = lists[0] if lists else []
            return data
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  Attempt {attempt} failed ({exc}). Retrying in {wait}s…",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"LLM returned unparseable response after {max_retries} attempts."
                ) from exc
        except Exception as exc:  # litellm wraps provider errors
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
    """Return True if the entry has all required fields with non-empty values."""
    required = {"category", "subcategory", "journey_stage", "issue_description", "resolution_steps"}
    return all(entry.get(k, "").strip() for k in required)


def _write_csv(path: Path, rows: list[dict], mode: str = "a") -> None:
    """Append (or write) rows to a CSV file."""
    write_header = mode == "w" or not path.exists() or path.stat().st_size == 0
    with path.open(mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KB_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic knowledge base entries using an LLM via litellm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        nargs="+",
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
        help="LLM sampling temperature (default: 0.8).",
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


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_path: Path = args.output or args.kb_path

    print(f"Model       : {args.model}")
    print(f"Count       : {args.count}")
    print(f"Batch size  : {args.batch_size}")
    print(f"KB path     : {args.kb_path}")
    print(f"Output path : {'[dry run]' if args.dry_run else output_path}")
    if args.categories:
        print(f"Categories  : {', '.join(args.categories)}")
    print()

    existing = _load_existing_kb(args.kb_path)
    print(f"Existing entries: {len(existing)}")

    targets = _pick_targets(args.categories, args.count, existing)
    if not targets:
        print("No new targets to generate — all category/subcategory combos already exist.")
        return

    actual_count = len(targets)
    if actual_count < args.count:
        print(
            f"Note: only {actual_count} unique unseen combinations available "
            f"(requested {args.count})."
        )

    next_id = _next_issue_id(existing)
    generated: list[dict] = []
    batches = [targets[i : i + args.batch_size] for i in range(0, len(targets), args.batch_size)]

    print(f"Generating {actual_count} entries in {len(batches)} batch(es)…\n")

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Batch {batch_num}/{len(batches)} ({len(batch)} entries)…", end=" ", flush=True)
        try:
            raw_entries = _generate_batch(
                targets=batch,
                existing=existing + generated,
                model=args.model,
                api_base=args.api_base,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
        except RuntimeError as exc:
            print(f"\nFailed: {exc}")
            continue

        for entry in raw_entries:
            if not _validate_entry(entry):
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
            next_id += 1

        print(f"OK (+{len(raw_entries)} entries)")

    if not generated:
        print("\nNo valid entries were generated.")
        return

    print(f"\nGenerated {len(generated)} valid entries.")

    if args.dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        writer = csv.DictWriter(sys.stdout, fieldnames=KB_COLUMNS)
        writer.writeheader()
        writer.writerows(generated)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if output_path != args.kb_path else "a"
        _write_csv(output_path, generated, mode=mode)
        print(f"Saved to: {output_path}")
        total = len(existing) + len(generated) if mode == "a" else len(generated)
        print(f"Total entries in KB: {total}")


if __name__ == "__main__":
    main()
