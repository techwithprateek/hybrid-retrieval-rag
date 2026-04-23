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
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw}") from exc
