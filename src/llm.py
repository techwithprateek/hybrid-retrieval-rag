"""
LLM Response Generator
Uses retrieved knowledge base context to generate a structured classification
and resolution recommendation for a customer support complaint.

How this fits in the RAG pipeline:
1. HybridRetriever (retrieval.py) finds the most relevant KB entries for a complaint.
2. This module formats those entries into a context string, sends them to the LLM
   together with the complaint, and parses/validates the structured JSON response.

Why RAG instead of just prompting the LLM directly?
- Without context, the LLM can hallucinate categories or steps that don't match
  the platform's actual policies and workflows.
- With the retrieved KB context, the LLM is *grounded* — it classifies and
  recommends based on real, verified knowledge base entries.
"""

import os
import json
from openai import OpenAI
import pandas as pd

# The OpenAI model used for classification and resolution generation.
# gpt-4o-mini balances cost, speed, and quality for structured JSON tasks.
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# System prompt — defines the LLM's persona, output format, and rules
# ---------------------------------------------------------------------------
# This is sent as the "system" message in the chat, which OpenAI models treat
# as a persistent instruction that frames every subsequent user message.
# The prompt is deliberately strict about JSON output to prevent the LLM from
# adding prose, markdown code fences, or other decorations around the JSON.
SYSTEM_PROMPT = """\
You are an expert customer support analyst. Your job is to classify a customer complaint \
and recommend the best resolution steps based on a provided knowledge base context.

You MUST respond with a single valid JSON object — no markdown, no code blocks, no extra text.

The JSON must follow this schema exactly:
{
  "category": "<main category of the issue>",
  "subcategory": "<specific sub-type of the issue>",
  "journey_stage": "<Pre-Purchase | Purchase | Post-Purchase>",
  "confidence": "<High | Medium | Low>",
  "summary": "<one-sentence summary of the issue>",
  "resolution_steps": ["<step 1>", "<step 2>", "..."]
}

Rules:
- Base your classification ONLY on the provided context.
- If the context is insufficient, use your best judgment and set confidence to "Low".
- resolution_steps must be an array of concise, actionable strings.
- Do not invent categories that are not present in the context.
"""


def build_context(retrieved: pd.DataFrame) -> str:
    """
    Format the retrieved knowledge base rows into a human-readable string
    that is injected into the LLM's user message as grounding context.

    Each section contains the full KB entry (category, subcategory, journey stage,
    issue description, resolution steps) plus the hybrid relevance score so the
    LLM can see how closely each entry matches the complaint.

    The sections are separated by "---" horizontal rules so the LLM can easily
    distinguish where one entry ends and the next begins.

    Args:
        retrieved: DataFrame returned by HybridRetriever.retrieve(), containing
                   the top-k KB rows with score columns attached.

    Returns:
        A multi-line string ready to be embedded in the user prompt.
    """
    sections = []
    for _, row in retrieved.iterrows():
        section = (
            f"Category: {row['category']}\n"
            f"Subcategory: {row['subcategory']}\n"
            f"Journey Stage: {row['journey_stage']}\n"
            f"Issue Description: {row['issue_description']}\n"
            f"Resolution Steps: {row['resolution_steps']}\n"
            f"Relevance Score: {row['hybrid_score']:.3f}"  # helps LLM weigh entries
        )
        sections.append(section)
    # Join sections with a clear delimiter so the LLM treats them as separate entries
    return "\n\n---\n\n".join(sections)


def generate_response(complaint: str, retrieved: pd.DataFrame) -> dict:
    """
    Core RAG generation step: send the complaint + retrieved KB context to the
    LLM and return a validated, structured classification dict.

    Flow:
    1. Validate that the API key is available.
    2. Format the retrieved KB entries into a context string.
    3. Build the user message combining the complaint and the context.
    4. Call the OpenAI chat completion API with JSON mode enforced.
    5. Parse the raw JSON string from the LLM response.
    6. Validate and coerce the parsed dict to the expected schema.
    7. Return the clean, type-safe dict to the caller (app.py).

    Args:
        complaint: The raw customer complaint text typed by the user.
        retrieved: DataFrame of top-k retrieved knowledge base entries,
                   as returned by HybridRetriever.retrieve().

    Returns:
        A dict with keys: category, subcategory, journey_stage, confidence,
        summary, resolution_steps (list of strings).

    Raises:
        ValueError: If the OpenAI API key is not configured or if the LLM
            response cannot be parsed as valid JSON.
    """
    # Fetch the API key from the environment.
    # app.py writes the sidebar key into os.environ["OPENAI_API_KEY"] before
    # calling this function, so this works whether the key came from .env or
    # the Streamlit sidebar.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please configure the OpenAI API key "
            "by setting the OPENAI_API_KEY environment variable before "
            "calling generate_response."
        )

    # Create an OpenAI client scoped to the current API key.
    # We create a new client per call (rather than a module-level singleton)
    # so that the correct key is always used, even if it changes between calls.
    client = OpenAI(api_key=api_key)

    # Format the retrieved KB entries into a string the LLM can read
    context = build_context(retrieved)

    # Build the user message with two clearly-labelled sections:
    # 1. The raw complaint text
    # 2. The KB context entries (the "retrieval" part of RAG)
    user_message = (
        f"## Customer Complaint\n{complaint}\n\n"
        f"## Knowledge Base Context\n{context}\n\n"
        "Now classify the complaint and provide resolution steps as JSON."
    )

    # Call the OpenAI Chat Completions API.
    # Key parameters:
    # - temperature=0.2: Low temperature makes the output deterministic and
    #   consistent — we want the same complaint to produce the same category
    #   every time, not creative variation.
    # - response_format={"type": "json_object"}: Tells the API to guarantee
    #   that the output is a valid JSON object.  This is more reliable than
    #   asking the model to "output JSON" in the prompt alone.
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},   # persistent instructions
            {"role": "user", "content": user_message},       # complaint + KB context
        ],
        temperature=0.2,                                     # low = consistent output
        response_format={"type": "json_object"},             # enforce valid JSON
    )

    # Extract the text content from the first (and only) completion choice
    raw = response.choices[0].message.content.strip()

    # Parse the JSON string — should always succeed due to response_format,
    # but we catch any edge cases and raise a descriptive error
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw}") from exc

    # Validate and coerce the parsed dict to the expected schema before
    # returning it to app.py — this protects the UI from type errors
    return _validate_and_coerce(result)


# ---------------------------------------------------------------------------
# Schema validation constants
# ---------------------------------------------------------------------------

# All fields that must exist in the LLM response as non-empty strings
_REQUIRED_STRING_KEYS = ("category", "subcategory", "journey_stage", "confidence", "summary")

# Allowed values for confidence (case-sensitive, as defined in the system prompt)
_VALID_CONFIDENCE = {"High", "Medium", "Low"}

# Allowed values for journey_stage (matches the KB's journey_stage column)
_VALID_JOURNEY_STAGES = {"Pre-Purchase", "Purchase", "Post-Purchase"}


def _validate_and_coerce(result: dict) -> dict:
    """
    Validate the LLM's parsed JSON response against the expected schema and
    coerce fields to the correct Python types where needed.

    Why is this necessary?
    - response_format={"type": "json_object"} only guarantees syntactically
      valid JSON — it does NOT guarantee the right keys or value types.
    - If the UI receives a malformed dict (e.g. resolution_steps is a string),
      it will crash when iterating over the steps.
    - This function acts as a safety net between the LLM and the UI.

    Validation rules:
    - All fields in _REQUIRED_STRING_KEYS must be present and non-empty strings.
    - resolution_steps must be coercible to a non-empty list of strings:
      * If it's already a list → filter out blank entries.
      * If it's a newline-delimited string → split on newlines (some models
        format steps as "1. Step one.\n2. Step two." instead of an array).
      * Any other type → raise a clear error.

    Raises:
        ValueError: If a required key is missing, blank, or resolution_steps
            cannot be coerced into a non-empty list.
    """
    # Guard: the top-level value must be a dict (JSON object), not a list or primitive
    if not isinstance(result, dict):
        raise ValueError(
            f"LLM response must be a JSON object, got {type(result).__name__}."
        )

    # Validate every required string field in one loop
    for key in _REQUIRED_STRING_KEYS:
        # Check the key exists at all
        if key not in result:
            raise ValueError(
                f"LLM response is missing required field '{key}'. "
                f"Full response: {result}"
            )
        # Check it's a non-empty string (not None, 0, empty string, or wrong type)
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(
                f"LLM response field '{key}' must be a non-empty string. "
                f"Got: {result[key]!r}"
            )
        # Strip surrounding whitespace from all string fields
        result[key] = result[key].strip()

    # --- Coerce resolution_steps to a list of non-empty strings ---
    steps = result.get("resolution_steps")

    if isinstance(steps, str):
        # Some models (especially local/smaller ones) return steps as a single
        # numbered string like "1. Check your UPI app.\n2. Retry after 5 minutes."
        # Split on newlines and keep non-blank lines.
        coerced = [s.strip() for s in steps.splitlines() if s.strip()]
        if not coerced:
            # Edge case: non-empty string with no newlines → treat as one step
            coerced = [steps.strip()]
        result["resolution_steps"] = coerced

    elif isinstance(steps, list):
        # Normal case: LLM returned a JSON array.
        # Convert each element to string and drop blank entries.
        coerced = [str(s).strip() for s in steps if str(s).strip()]
        if not coerced:
            raise ValueError(
                "LLM response 'resolution_steps' list is empty or contains only blank entries."
            )
        result["resolution_steps"] = coerced

    else:
        # Unexpected type (dict, int, None, …) — we cannot safely coerce this
        raise ValueError(
            f"LLM response 'resolution_steps' must be a list or string, "
            f"got {type(steps).__name__!r}. Full response: {result}"
        )

    return result
