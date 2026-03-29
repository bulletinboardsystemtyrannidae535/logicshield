"""
LogicShield - JSON Structure Repair Module
==========================================
Deterministic repair of malformed JSON from LLM responses.

LLMs frequently produce syntactically broken JSON (trailing commas,
unquoted keys, markdown fences, single quotes). This module repairs
common failures without altering semantic meaning.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def repair_json(raw: str) -> dict:
    """
    Attempt to parse and repair malformed JSON from an LLM response.
    
    Repair steps (applied in order):
        1. Strip markdown code fences
        2. Extract JSON object from surrounding text
        3. Try standard json.loads
        4. Fix single quotes -> double quotes
        5. Fix trailing commas
        6. Fix unquoted keys
        7. Strip comments
    
    Args:
        raw: Raw string response from an LLM
        
    Returns:
        Parsed dict
        
    Raises:
        ValueError: If JSON cannot be repaired
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from LLM.")

    text = raw.strip()

    # Step 1: Strip markdown code fences
    text = _strip_code_fences(text)

    # Step 2: Extract JSON object from surrounding text
    text = _extract_json_object(text)

    # Step 3: Try standard parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4: Fix single quotes
    repaired = _fix_single_quotes(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 5: Fix trailing commas
    repaired = _fix_trailing_commas(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 6: Fix unquoted keys
    repaired = _fix_unquoted_keys(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 7: Strip inline comments
    repaired = _strip_comments(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse or repair JSON from LLM response. "
            f"Error: {e}. Raw response (first 200 chars): {raw[:200]}"
        )


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    # Match ```json\n...\n``` or ```\n...\n```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object {...} from surrounding text."""
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _fix_single_quotes(text: str) -> str:
    """Replace single quotes with double quotes (naive but effective)."""
    # Only replace quotes that look like JSON delimiters
    # This is imperfect but handles the 90% case from LLMs
    return text.replace("'", '"')


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_unquoted_keys(text: str) -> str:
    """Quote unquoted keys: {key: value} -> {"key": value}."""
    return re.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', text)


def _strip_comments(text: str) -> str:
    """Remove // inline comments and # comments."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Remove // comments (but not inside strings -- best effort)
        line = re.sub(r'//.*$', '', line)
        line = re.sub(r'#.*$', '', line)
        cleaned.append(line)
    return "\n".join(cleaned)
