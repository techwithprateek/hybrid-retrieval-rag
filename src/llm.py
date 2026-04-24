"""
LLM Response Generator
Uses retrieved knowledge base context to generate a structured classification
and resolution recommendation for a customer support complaint.
"""

import os
import json
from openai import OpenAI
import pandas as pd

LLM_MODEL = "gpt-4o-mini"

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
    """Format retrieved knowledge base rows into a readable context string."""
    sections = []
    for _, row in retrieved.iterrows():
        section = (
            f"Category: {row['category']}\n"
            f"Subcategory: {row['subcategory']}\n"
            f"Journey Stage: {row['journey_stage']}\n"
            f"Issue Description: {row['issue_description']}\n"
            f"Resolution Steps: {row['resolution_steps']}\n"
            f"Relevance Score: {row['hybrid_score']:.3f}"
        )
        sections.append(section)
    return "\n\n---\n\n".join(sections)


def generate_response(complaint: str, retrieved: pd.DataFrame) -> dict:
    """
    Generate a structured classification and resolution for a customer complaint.

    Args:
        complaint: The raw customer complaint text.
        retrieved: DataFrame of top-k retrieved knowledge base entries.

    Returns:
        A dict with keys: category, subcategory, journey_stage, confidence,
        summary, resolution_steps.

    Raises:
        ValueError: If the OpenAI API key is not configured or if the LLM
            response cannot be parsed as valid JSON.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please configure the OpenAI API key "
            "by setting the OPENAI_API_KEY environment variable before "
            "calling generate_response."
        )

    client = OpenAI(api_key=api_key)
    context = build_context(retrieved)

    user_message = (
        f"## Customer Complaint\n{complaint}\n\n"
        f"## Knowledge Base Context\n{context}\n\n"
        "Now classify the complaint and provide resolution steps as JSON."
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw}") from exc

    return _validate_and_coerce(result)


_REQUIRED_STRING_KEYS = ("category", "subcategory", "journey_stage", "confidence", "summary")
_VALID_CONFIDENCE = {"High", "Medium", "Low"}
_VALID_JOURNEY_STAGES = {"Pre-Purchase", "Purchase", "Post-Purchase"}


def _validate_and_coerce(result: dict) -> dict:
    """
    Validate and coerce the LLM output to the expected schema.

    Ensures all required keys are present with non-empty string values,
    normalises *confidence* and *journey_stage* to their canonical forms,
    and guarantees *resolution_steps* is a list of non-empty strings.

    Raises:
        ValueError: If a required key is missing or *resolution_steps* cannot
            be coerced into a non-empty list.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"LLM response must be a JSON object, got {type(result).__name__}."
        )

    # Validate required string fields
    for key in _REQUIRED_STRING_KEYS:
        if key not in result:
            raise ValueError(
                f"LLM response is missing required field '{key}'. "
                f"Full response: {result}"
            )
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(
                f"LLM response field '{key}' must be a non-empty string. "
                f"Got: {result[key]!r}"
            )
        result[key] = result[key].strip()

    # Coerce resolution_steps to a list of non-empty strings
    steps = result.get("resolution_steps")
    if isinstance(steps, str):
        # Some models return the steps as a single numbered string — split on newlines
        coerced = [s.strip() for s in steps.splitlines() if s.strip()]
        if not coerced:
            # Fall back: treat the whole string as a single step
            coerced = [steps.strip()]
        result["resolution_steps"] = coerced
    elif isinstance(steps, list):
        coerced = [str(s).strip() for s in steps if str(s).strip()]
        if not coerced:
            raise ValueError(
                "LLM response 'resolution_steps' list is empty or contains only blank entries."
            )
        result["resolution_steps"] = coerced
    else:
        raise ValueError(
            f"LLM response 'resolution_steps' must be a list or string, "
            f"got {type(steps).__name__!r}. Full response: {result}"
        )

    return result
