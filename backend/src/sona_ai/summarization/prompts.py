import json
import random
from typing import Optional

DEFAULT_SUMMARY_PLAN = {
    "recording_type": "general conversation",
    "format_name": "General Summary",
    "audience": "general reader",
    "sections": ["Overview", "Key Points", "Important Details"],
    "style": "concise, factual, neutral",
    "rationale": "Default format used when no more specific format is detected.",
}

PROMPT_VARIATIONS = [
    (
        "You are a professional news editor.\n"
        "Write a concise and coherent summary of the following conversation.\n"
        "Focus on the key facts and avoid unnecessary details.\n\n"
    ),
    (
        "Summarize the following conversation in 3-5 sentences.\n"
        "The summary should be factual, concise, and written in a neutral tone.\n\n"
    ),
    (
        "You are a news editor.\n"
        "Write a concise news-style summary of the following broadcast conversation.\n"
        "Include the key events and important statements.\n"
        "Do not repeat information.\n\n"
    ),
]

def parse_summary_plan(raw: str) -> dict:
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_SUMMARY_PLAN.copy()

    if not isinstance(plan, dict):
        return DEFAULT_SUMMARY_PLAN.copy()

    default = DEFAULT_SUMMARY_PLAN.copy()
    default.update({
        key: value
        for key, value in plan.items()
        if key in default and value
    })

    if not isinstance(default.get("sections"), list):
        default["sections"] = DEFAULT_SUMMARY_PLAN["sections"]

    default["sections"] = [
        str(section).strip()
        for section in default["sections"]
        if str(section).strip()
    ] or DEFAULT_SUMMARY_PLAN["sections"]

    return default


def build_prompt(transcript: str, prompt: Optional[str] = None) -> str:
    instruction = prompt if prompt else random.choice(PROMPT_VARIATIONS)
    return f"{instruction}\n\nTranscript:\n{transcript}\n\nSummary:"

def build_summary_planner_prompt(
    transcript: str,
    user_instruction: Optional[str] = None,
) -> str:
    instruction = user_instruction or "No extra user instruction."
    return (
        "Analyze the recording transcript and choose the best summary format.\n"
        "Return JSON only. Do not include markdown. Do not include chain-of-thought.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "recording_type": "interview | meeting | lecture | podcast | user research | sales call | general conversation",\n'
        '  "format_name": "short human-readable format name",\n'
        '  "audience": "intended reader",\n'
        '  "sections": ["section heading 1", "section heading 2"],\n'
        '  "style": "tone and writing style",\n'
        '  "rationale": "one short sentence explaining the format choice"\n'
        "}\n\n"
        f"User instruction:\n{instruction}\n\n"
        f"Transcript:\n{transcript}\n"
    )

def build_adaptive_summary_prompt(
    transcript: str,
    plan: dict,
    user_instruction: Optional[str] = None,
) -> str:
    instruction = user_instruction or "No extra user instruction."
    sections = plan.get("sections") or DEFAULT_SUMMARY_PLAN["sections"]
    section_text = "\n".join(f"- {section}" for section in sections)

    return (
        "Write a summary of the recording transcript using the provided summary plan.\n"
        "Use Markdown formatting.\n"
        "Use section headings for major sections.\n"
        "Use bullet lists when they improve readability.\n"
        "Use bold text sparingly for important labels, names, or decisions.\n"
        "Do not wrap the summary in a markdown code fence.\n"
        "Do not output raw HTML.\n"
        "Use only information supported by the transcript.\n"
        "Do not invent facts. Do not mention any summary plan.\n\n"
        "Do not add an overall title.\n"
        "Start directly with the first required section heading.\n"
        "Use only the required sections as visible headings.\n"
        f"Internal format label, do not include in output: {plan.get('format_name', 'General Summary')}\n"
        f"Recording type: {plan.get('recording_type', 'general conversation')}\n"
        f"Audience: {plan.get('audience', 'general reader')}\n"
        f"Style: {plan.get('style', 'concise, factual, neutral')}\n"
        f"Required sections:\n{section_text}\n\n"
        f"User instruction:\n{instruction}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Summary:"
    )