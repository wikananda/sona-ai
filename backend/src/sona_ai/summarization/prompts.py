import json
import random
import re
import textwrap
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


def parse_adaptive_summary_response(raw: str) -> dict:
    cleaned = _strip_json_fence(raw.strip())
    tagged = _parse_tagged_adaptive_response(cleaned)
    if tagged is not None:
        return tagged

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _parse_malformed_adaptive_json(cleaned)
        if extracted is not None:
            return extracted

        return {
            "summary": raw.strip(),
            "plan": DEFAULT_SUMMARY_PLAN.copy(),
            "format_name": DEFAULT_SUMMARY_PLAN["format_name"],
        }

    if not isinstance(payload, dict):
        return {
            "summary": raw.strip(),
            "plan": DEFAULT_SUMMARY_PLAN.copy(),
            "format_name": DEFAULT_SUMMARY_PLAN["format_name"],
        }

    plan = parse_summary_plan(json.dumps(payload.get("plan", {})))
    summary = str(payload.get("summary_markdown") or payload.get("summary") or "").strip()
    if not summary:
        summary = raw.strip()
    summary = _normalize_summary_markdown(summary, plan)

    return {
        "summary": summary,
        "plan": plan,
        "format_name": plan.get("format_name"),
    }


def _parse_tagged_adaptive_response(raw: str) -> Optional[dict]:
    plan_text = _extract_tag(raw, "plan_json")
    summary = _extract_tag(raw, "summary_markdown")
    if plan_text is None:
        return None

    if summary is None:
        summary = _extract_after_tag(raw, "plan_json")
    if summary is None:
        summary = ""

    plan = parse_summary_plan(plan_text)
    summary = _normalize_summary_markdown(summary, plan)
    return {
        "summary": summary,
        "plan": plan,
        "format_name": plan.get("format_name"),
    }


def _extract_tag(raw: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return textwrap.dedent(match.group(1)).strip()


def _extract_after_tag(raw: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"</{tag}>",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return textwrap.dedent(raw[match.end():]).strip()


def _parse_malformed_adaptive_json(raw: str) -> Optional[dict]:
    plan_text = _extract_json_object_after_key(raw, "plan")
    summary = _extract_string_after_key(raw, "summary_markdown")
    if plan_text is None and summary is None:
        return None

    plan = parse_summary_plan(plan_text or "{}")
    summary = _normalize_summary_markdown(summary or raw.strip(), plan)
    return {
        "summary": summary,
        "plan": plan,
        "format_name": plan.get("format_name"),
    }


def _extract_json_object_after_key(raw: str, key: str) -> Optional[str]:
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*{{', raw)
    if key_match is None:
        return None

    start = raw.find("{", key_match.start())
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start:index + 1]

    return None


def _extract_string_after_key(raw: str, key: str) -> Optional[str]:
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw)
    if key_match is None:
        return None

    start = key_match.end()
    end_match = re.search(r'"\s*}\s*$', raw[start:], flags=re.DOTALL)
    end = start + end_match.start() if end_match is not None else raw.rfind('"')
    if end <= start:
        return None

    summary = textwrap.dedent(raw[start:end]).strip()
    summary = summary.removesuffix("}").strip()
    return summary


def _normalize_summary_markdown(summary: str, plan: dict) -> str:
    normalized = _strip_markdown_fence(textwrap.dedent(summary).strip())
    normalized = _strip_adaptive_tags(normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    normalized = _normalize_bold_heading_lines(normalized)
    sections = plan.get("sections") or []
    for section in sections:
        section_text = str(section).strip()
        if not section_text:
            continue

        escaped = re.escape(section_text)
        normalized = re.sub(
            rf"(?m)^#{{1,6}}\s+{escaped}\s+(\S)",
            rf"## {section_text}\n\1",
            normalized,
        )
        normalized = re.sub(
            rf"(?m)^{escaped}\s+(\S)",
            rf"## {section_text}\n\1",
            normalized,
        )
        normalized = re.sub(
            rf"(?m)^{escaped}\s*$",
            f"## {section_text}",
            normalized,
        )

    normalized = _ensure_heading_spacing(normalized)
    return normalized.strip()


def _normalize_bold_heading_lines(raw: str) -> str:
    return re.sub(
        r"(?m)^\*\*([^*\n]+)\*\*\s*$",
        r"## \1",
        raw,
    )


def _ensure_heading_spacing(raw: str) -> str:
    normalized = re.sub(
        r"(?m)^(#{1,6}\s+[^\n]+)\n(?!\n)",
        r"\1\n\n",
        raw,
    )
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _strip_adaptive_tags(raw: str) -> str:
    return re.sub(
        r"</?(?:plan_json|summary_markdown)>",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()


def _strip_markdown_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw

    lines = raw.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return raw


def _strip_json_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw

    lines = raw.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return raw


def build_prompt(transcript: str, prompt: Optional[str] = None) -> str:
    instruction = prompt if prompt else random.choice(PROMPT_VARIATIONS)
    return f"{instruction}\n\nTranscript:\n{transcript}\n\nSummary:"


def build_one_call_adaptive_summary_prompt(
    transcript: str,
    user_instruction: Optional[str] = None,
) -> str:
    instruction = user_instruction or "No extra user instruction."
    return (
        "Analyze the recording transcript, choose the best summary format, and "
        "write the final summary in one response.\n"
        "Return only the tagged envelope below. Both tags are required. Do not "
        "include chain-of-thought.\n"
        "Do not put summary text outside <summary_markdown>.\n\n"
        "<plan_json>\n"
        "{\n"
        '  "recording_type": "interview | meeting | lecture | podcast | user research | sales call | general conversation",\n'
        '  "format_name": "short human-readable format name",\n'
        '  "audience": "intended reader",\n'
        '  "sections": ["section heading 1", "section heading 2"],\n'
        '  "style": "tone and writing style",\n'
        '  "rationale": "one short sentence explaining the format choice"\n'
        "}\n"
        "</plan_json>\n\n"
        "<summary_markdown>\n"
        "Final user-visible Markdown summary.\n"
        "</summary_markdown>\n\n"
        "Summary rules:\n"
        "- Use Markdown formatting inside <summary_markdown> only.\n"
        "- Use level-2 Markdown headings for every major section, like: ## Section Heading.\n"
        "- Never use bold text as a section heading. Do not write **Section Heading**.\n"
        "- Put one blank line after every heading.\n"
        "- Under each heading, write either one concise paragraph or 2-4 bullet points.\n"
        "- Do not mix loose paragraph text and bullets under the same heading unless necessary.\n"
        "- Use bold text only inside paragraphs or bullets for important labels, names, or decisions.\n"
        "- Do not add an overall title.\n"
        "- Start directly with the first section heading.\n"
        "- Use only information supported by the transcript.\n"
        "- Do not invent facts. Do not mention the summary plan.\n\n"
        f"User instruction:\n{instruction}\n\n"
        f"Transcript:\n{transcript}\n"
    )

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
