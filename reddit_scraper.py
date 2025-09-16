# reddit_scraper.py
import os
from dotenv import load_dotenv

import pandas as pd
import praw

load_dotenv()

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ----------------------- helpers -----------------------
BAD_BODIES = {"[deleted]", "[removed]"}

def _is_valid_body(text: str) -> bool:
    if text is None:
        return False
    s = str(text).strip()
    if not s:
        return False
    return s.lower() not in BAD_BODIES

def _collect_from_submission(submission, valid_needed: int) -> list[dict]:
    """
    From a single submission, collect up to `valid_needed` VALID comments
    (skipping [deleted]/[removed]/empty).
    """
    collected: list[dict] = []

    # Expand 'more' once; we’ll break as soon as we have enough valid comments
    submission.comments.replace_more(limit=0)

    for c in submission.comments.list():
        if len(collected) >= valid_needed:
            break

        body = c.body if hasattr(c, "body") else None
        if not _is_valid_body(body):
            continue

        collected.append({
            "post_title": submission.title,
            "author": str(c.author) if getattr(c, "author", None) is not None else "None",
            "score": getattr(c, "score", 0),
            "body": body.strip(),
            "created_utc": getattr(c, "created_utc", None),
        })

    return collected

# ----------------------- main -----------------------
def scrape_reddit(
    subreddit: str | None = None,
    post_limit: int = 25,
    comments_per_post: int = 10,
    post_url: str | None = None,
    keywords: list[str] | None = None,
) -> pd.DataFrame:
    """
    Scrape Reddit comments and return ONLY valid comments
    (no [deleted]/[removed]/empty). Attempts to return exactly
    `comments_per_post` comments per post when possible.

    Returns columns: ['post_title', 'author', 'score', 'body', 'created_utc']
    """
    all_comments: list[dict] = []

    # ---------- Specific post ----------
    if post_url:
        submission = reddit.submission(url=post_url)
        all_comments.extend(_collect_from_submission(submission, comments_per_post))

    # ---------- Posts by keywords ----------
    elif keywords and subreddit:
        seen_ids = set()
        for kw in keywords:
            for submission in reddit.subreddit(subreddit).search(kw, limit=post_limit):
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)
                all_comments.extend(_collect_from_submission(submission, comments_per_post))

    # ---------- Top/hot posts from subreddit ----------
    elif subreddit:
        for submission in reddit.subreddit(subreddit).hot(limit=post_limit):
            all_comments.extend(_collect_from_submission(submission, comments_per_post))

    else:
        raise ValueError("Either subreddit or post_url must be provided.")

    df = pd.DataFrame(all_comments)

    # Ensure expected columns exist even if empty (to avoid downstream key errors)
    expected_cols = ["post_title", "author", "score", "body", "created_utc"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object" if col in ("post_title", "author", "body") else "float64")

    # Cast created_utc to numeric if present (useful for later to_datetime(unit='s'))
    if not df.empty and "created_utc" in df.columns:
        df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")

    return df
