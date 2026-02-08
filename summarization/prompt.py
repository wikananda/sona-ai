def build_prompt(transcript: str) -> str:
    """Helper to prompt format the input for MediaSum"""
    return (
        "Summarize the following conversation into a concise, well-written summary:\n\n"
        f"{transcript}\n\n"
        "Summary:"
    )