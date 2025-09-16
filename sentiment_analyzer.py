# sentiment_analyzer.py
import os
import re
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _normalize_sentiment(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("pos"): return "Positive"
    if s.startswith("neg"): return "Negative"
    return "Neutral"

def analyze_sentiment_gpt(
    comments: List[str],
    model: str = "gpt-3.5-turbo",
    batch_size: int = 40,
) -> List[str]:
    """
    Returns a list of sentiments (strings) aligned to input length.
    Each item is one of: 'Positive', 'Neutral', 'Negative'.
    """
    n = len(comments)
    if n == 0:
        return []

    # Initialize all as Neutral; we'll overwrite when we parse model output
    sentiments = ["Neutral"] * n

    # Process in batches to avoid token limits
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = comments[start:end]

        # Number lines so the model can return `<index>|<sentiment>`
        numbered = []
        for i, c in enumerate(batch, start=1):
            text = (c or "").replace("\n", " ").strip()
            if len(text) > 700:
                text = text[:700] + "…"
            numbered.append(f"{start + i}. {text}")  # global index (1-based)

        prompt = (
            "Act as marketing Specialist and Classify each comment as Positive, Neutral, or Negative.\n"
            "For EACH line below, reply with a single line in the exact format:\n"
            "<index>|<sentiment>\n\n"
            "Comments:\n" + "\n".join(numbered)
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise marketing analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            out = resp.choices[0].message.content.strip().splitlines()
        except Exception as e:
            # On API error, keep Neutral for this batch and continue
            continue

        # Parse lines like: "17|Positive" or "17 - Negative" or "17: Neutral"
        for line in out:
            m = re.match(r"\s*(\d+)\s*[\|\-:]\s*([A-Za-z]+)", line)
            if not m:
                continue
            idx_1 = int(m.group(1))  # 1-based index we sent
            idx_0 = idx_1 - 1        # back to 0-based
            if 0 <= idx_0 < n:
                sentiments[idx_0] = _normalize_sentiment(m.group(2))

    return sentiments
