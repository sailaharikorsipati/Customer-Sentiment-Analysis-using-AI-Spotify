# app.py
import os
from datetime import datetime
from collections import Counter
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

from reddit_scraper import scrape_reddit
from sentiment_analyzer import analyze_sentiment_gpt
from summarizer import summarize_comments

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")
st.title("🎧 Reddit Sentiment Analyzer — Spotify Insights")
st.markdown(
    "<p style='text-align: center; color: grey; font-size:18px;'>"
    "From consumer chatter to strategic clarity — AI-powered insights for your next big move."
    "</p>",
    unsafe_allow_html=True
)


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize session_state
for k, v in {
    "df": None,
    "comments": [],
    "cleaned_comments": [],
    "comment_embs": None,
    "qa_answer": None,
    "qa_sources": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    subreddit = st.text_input("Subreddit", "spotify")
    post_url = st.text_input("Specific Reddit post URL (optional)")
    keywords_input = st.text_input("Keywords (comma-separated, optional)")
    post_limit = st.slider("Posts to fetch", 5, 100, 25)
    comments_per_post = st.slider("Comments per post", 1, 50, 10)

    use_timeline = st.checkbox("Use timeline filter?", value=False)
    if use_timeline:
        from_date = st.date_input("From", value=datetime(2025, 1, 1))
        to_date = st.date_input("To", value=datetime.today())
    else:
        from_date, to_date = None, None

    run = st.button("Run Analysis")

# ---------- Helpers ----------
def run_gpt_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Call GPT once to label ALL rows with Positive/Neutral/Negative."""
    if "sentiment" in df.columns:
        return df
    try:
        with st.spinner("Analyzing sentiment with GPT..."):
            sentiments = analyze_sentiment_gpt(df["body"].tolist())
        df["sentiment"] = [
            s if s in ("Positive", "Neutral", "Negative") else "Neutral"
            for s in sentiments
        ]
    except Exception as e:
        st.error(f"Error in GPT sentiment analysis: {e}")
    return df

def _batched(it, n=96):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def get_embeddings(texts, model=EMBED_MODEL):
    """Return np.ndarray [N, D] embeddings."""
    embs = []
    for batch in _batched(texts, n=96):
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs, dtype="float32")

def cosine_sim_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

def clean_comments(raw, max_chars=700):
    cleaned = []
    for c in raw:
        s = (c or "")
        s = s.replace("\n", " ").strip()
        if len(s) > max_chars:
            s = s[:max_chars] + "…"
        cleaned.append(s)
    return cleaned

def top_k_with_openai(question, comment_embs, cleaned_comments, k=15):
    q_emb = get_embeddings([question])
    sims = cosine_sim_matrix(q_emb, comment_embs).ravel()
    idx = sims.argsort()[::-1][:k]
    return [(cleaned_comments[i], float(sims[i])) for i in idx]

def answer_over_selected(question, selected):
    if not selected:
        return "No relevant comments found to answer this question."
    context = "\n".join(f"- {txt}" for (txt, _sim) in selected)
    prompt = (
        "You are a precise marketing insights analyst. Answer the user's question using ONLY the comments below. "
        f"COMMENTS:\n{context}\n\nQUESTION: {question}\n\n"
        "Provide a concise, bullet-point answer. If applicable, call out clear pros, cons, or themes."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You ground every answer in the provided comments only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

# ---------- Feature extraction (Spotify only) ----------
def extract_themes_from_comments(comments: list[str],
                                 model: str = "gpt-3.5-turbo",
                                 batch_size: int = 40,
                                 max_themes_per_comment: int = 2) -> dict[int, list[str]]:
    """Extract up to N short 'themes' per comment via GPT; returns {idx: [themes]}."""
    if not comments:
        return {}
    themes_by_idx: dict[int, list[str]] = {}
    for start in range(0, len(comments), batch_size):
        end = min(start + batch_size, len(comments))
        numbered = []
        for i, c in enumerate(comments[start:end], start=1):
            txt = (c or "").replace("\n", " ").strip()
            if len(txt) > 700:
                txt = txt[:700] + "…"
            numbered.append(f"{start + i}. {txt}")
        prompt = (
            f"Extract up to {max_themes_per_comment} SHORT feature 'themes' (2-3 words) from each comment. "
            "Focus on features like UI, playlists, recommendations, pricing, quality. "
            "Return one line per comment as <index>|theme1, theme2"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a marketing text-mining assistant."},
                    {"role": "user", "content": prompt + "\n\nComments:\n" + "\n".join(numbered)},
                ],
                temperature=0,
            )
            lines = resp.choices[0].message.content.strip().splitlines()
        except Exception:
            lines = []
        for line in lines:
            if "|" not in line:
                continue
            left, right = line.split("|", 1)
            if not left.strip().isdigit():
                continue
            idx0 = int(left.strip()) - 1
            if not (0 <= idx0 < len(comments)):
                continue
            raw_themes = [t.strip() for t in right.split(",") if t.strip()]
            cleaned = []
            for t in raw_themes:
                norm = re.sub(r"\s+", " ", t.lower()).strip()
                norm = norm.title()
                norm = norm.replace("Ui", "UI").replace("Ux", "UX")
                cleaned.append(norm)
            themes_by_idx[idx0] = cleaned
    return themes_by_idx

def build_spotify_theme_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return table: Sentiment | Features (with counts) | Distinct Feature Count (comments mentioning features)."""
    if df.empty or "body" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["Sentiment", "Features (with counts)", "Distinct Feature Count"])

    comments = df["body"].fillna("").tolist()
    sentiments = df["sentiment"].astype(str).str.title().tolist()
    themes_by_idx = extract_themes_from_comments(comments)

    counts = {"Positive": Counter(), "Negative": Counter()}
    comment_tracker = {"Positive": set(), "Negative": set()}

    for i, sent in enumerate(sentiments):
        themes = themes_by_idx.get(i, [])
        if sent in ("Positive", "Negative") and themes:
            comment_tracker[sent].add(i)
            for t in themes:
                counts[sent][t] += 1

    rows = []
    for sent in ["Positive", "Negative"]:
        ctr = counts[sent]
        total_comments = len(comment_tracker[sent])
        features = ", ".join([f"{k} ({v})" for k, v in ctr.most_common(8)]) if total_comments else ""
        rows.append({
            "Sentiment": sent,
            "Features (with counts)": features,
            "Distinct Feature Count": total_comments
        })

    return pd.DataFrame(rows)

def build_feature_sentiment_matrix(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Build a matrix of Feature | Positive | Negative | Total
    using the GPT-extracted themes. Returns top_n features by Total.
    """
    if df.empty or "body" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["Feature", "Positive", "Negative", "Total"])

    comments = df["body"].fillna("").tolist()
    sentiments = df["sentiment"].astype(str).str.title().tolist()

    themes_by_idx = extract_themes_from_comments(comments)

    pos_counts = {}
    neg_counts = {}

    for i, sent in enumerate(sentiments):
        themes = themes_by_idx.get(i, [])
        for t in themes:
            if sent == "Positive":
                pos_counts[t] = pos_counts.get(t, 0) + 1
            elif sent == "Negative":
                neg_counts[t] = neg_counts.get(t, 0) + 1

    all_feats = set(pos_counts) | set(neg_counts)
    rows = []
    for feat in all_feats:
        p = pos_counts.get(feat, 0)
        n = neg_counts.get(feat, 0)
        rows.append({"Feature": feat, "Positive": p, "Negative": n, "Total": p + n})

    if not rows:
        return pd.DataFrame(columns=["Feature", "Positive", "Negative", "Total"])

    mat = pd.DataFrame(rows).sort_values(["Total", "Positive"], ascending=[False, False])
    return mat.head(top_n).reset_index(drop=True)

#This is for the stacked bar chart
def build_weighted_feature_matrix(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return long-format df with columns:
    Feature | sentiment | upvotes
    where upvotes is the sum(score) for comments that mentioned the feature with that sentiment.
    """
    needed = {"body", "sentiment", "score"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame(columns=["Feature", "sentiment", "upvotes"])

    tmp = df.copy()
    tmp["sentiment"] = tmp["sentiment"].astype(str).str.title()
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce").fillna(0)

    # Extract themes per comment using your existing GPT extractor
    themes_by_idx = extract_themes_from_comments(tmp["body"].fillna("").tolist())

    # Aggregate weights
    pos_w = {}
    neg_w = {}
    for i, row in tmp.reset_index(drop=True).iterrows():
        themes = themes_by_idx.get(i, [])
        if not themes:
            continue
        s = row["sentiment"]
        w = float(row["score"])
        for t in themes:
            if s == "Positive":
                pos_w[t] = pos_w.get(t, 0.0) + w
            elif s == "Negative":
                neg_w[t] = neg_w.get(t, 0.0) + w

    # Build table, keep top_n by total weight
    feats = set(pos_w) | set(neg_w)
    rows = []
    for f in feats:
        p = pos_w.get(f, 0.0)
        n = neg_w.get(f, 0.0)
        rows.append({"Feature": f, "Positive": p, "Negative": n, "Total": p + n})

    if not rows:
        return pd.DataFrame(columns=["Feature", "sentiment", "upvotes"])

    wide = pd.DataFrame(rows).sort_values("Total", ascending=False).head(top_n)

    # Long format for stacked bar
    long = pd.melt(
        wide[["Feature", "Positive", "Negative"]],
        id_vars=["Feature"],
        var_name="sentiment",
        value_name="upvotes",
    )
    return long


# ---------- Run Analysis ----------
if run:
    st.session_state["qa_answer"] = None
    st.session_state["qa_sources"] = []
    keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input else None

    with st.spinner("Fetching Reddit comments..."):
        try:
            df = scrape_reddit(
                subreddit=subreddit if not post_url else None,
                post_limit=post_limit,
                comments_per_post=comments_per_post,
                post_url=post_url if post_url else None,
                keywords=keywords,
            )
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            df = pd.DataFrame()

    if df.empty:
        st.warning("No comments fetched.")
    else:
        # Convert timestamp to datetime and optionally filter by timeline
        if "created_utc" in df.columns:
            df["created_date"] = pd.to_datetime(df["created_utc"], unit="s")
        if from_date and to_date and "created_date" in df.columns:
            df = df[(df["created_date"].dt.date >= from_date) & (df["created_date"].dt.date <= to_date)]

        if not df.empty:
            # Sentiment labels
            df = run_gpt_sentiment(df)

            # Persist for rest of app
            st.session_state["df"] = df
            comments = df["body"].fillna("").tolist()

            # Cap for embeddings to keep responsive & cost-effective
            max_for_embed = 350
            if len(comments) > max_for_embed:
                comments = comments[:max_for_embed]

            st.session_state["comments"] = comments
            st.session_state["cleaned_comments"] = clean_comments(comments)

            # Build embeddings (one-time per run)
            with st.spinner("Indexing comments with embeddings..."):
                try:
                    st.session_state["comment_embs"] = get_embeddings(st.session_state["cleaned_comments"])
                except Exception as e:
                    st.error(f"Embeddings failed: {e}")
                    st.session_state["comment_embs"] = None
        else:
            st.session_state["df"] = None

# ---------- Main Panels ----------
df = st.session_state["df"]
if df is not None and not df.empty:
    # Equal widths
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sentiment Distribution")
        df_plot = df.copy()
        df_plot["sentiment"] = df_plot["sentiment"].astype(str).str.title()
        counts = (
            df_plot["sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            .reset_index()
        )
        counts.columns = ["sentiment", "count"]
        color_scale = alt.Scale(
            domain=["Positive", "Neutral", "Negative"],
            range=["#2ecc71", "#95a5a6", "#e74c3c"]
        )
        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("sentiment:N", title="Sentiment"),
                y=alt.Y("count:Q", title="Number of Comments"),
                color=alt.Color("sentiment:N", scale=color_scale),
                tooltip=["sentiment", "count"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Ask Questions about the Comments")
        user_question = st.text_area(
            "Ask about features, complaints, pricing, recommendations…",
            "",
            key="qa_box",
            height=120,
        )
        get_answer = st.button("Get Answer")
        if get_answer and user_question.strip():
            if not st.session_state["comments"]:
                st.warning("No comments available.")
            elif st.session_state["comment_embs"] is None:
                st.warning("Embeddings not ready.")
            else:
                with st.spinner("Finding relevant comments..."):
                    selected = top_k_with_openai(
                        user_question,
                        st.session_state["comment_embs"],
                        st.session_state["cleaned_comments"],
                        k=15,
                    )
                    ans = answer_over_selected(user_question, selected)
                    st.session_state["qa_answer"] = ans
                    st.session_state["qa_sources"] = selected
        if st.session_state["qa_answer"]:
            st.markdown(st.session_state["qa_answer"])
            with st.expander("Supporting comments"):
                src_df = pd.DataFrame(
                    [{"similarity": round(sim, 3), "comment": txt} for (txt, sim) in (st.session_state["qa_sources"] or [])]
                )
                st.dataframe(src_df.reset_index(drop=True))

    st.subheader("Top Features — Weighted Sentiment (Sum of Upvotes)")
    wf = build_weighted_feature_matrix(df, top_n=10)
    if wf.empty:
        st.info("No weighted feature data yet.")
    else:
        color_scale_feat = alt.Scale(domain=["Positive", "Negative"], range=["#2ecc71", "#e74c3c"])
        stacked = (
            alt.Chart(wf)
            .mark_bar()
            .encode(
                x=alt.X("Feature:N", sort="-y", title="Feature"),
                y=alt.Y("upvotes:Q", title="Sum of Upvotes"),
                color=alt.Color("sentiment:N", scale=color_scale_feat, title="Sentiment"),
                tooltip=[
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("upvotes:Q", title="Upvotes (sum)"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(stacked, use_container_width=True)


    # Feature × Sentiment Matrix
    st.subheader("Feature × Sentiment Matrix")
    with st.spinner("Mining top features and their sentiment…"):
        matrix_df = build_feature_sentiment_matrix(df, top_n=20)

    if matrix_df.empty:
        st.info("No feature mentions detected yet.")
    else:
        clean_matrix = matrix_df[["Feature", "Positive", "Negative", "Total"]].reset_index(drop=True)
        st.dataframe(clean_matrix, use_container_width=True,hide_index=True)
        #st.download_button(
            #"Download Feature × Sentiment Matrix (CSV)",
            #clean_matrix.to_csv(index=False),
            #file_name="feature_sentiment_matrix.csv",
        #)


    # ---------- Trend: Positive & Negative comment counts by month (ONE line per sentiment) ----------
    if "created_date" in df.columns:
        st.subheader("Sentiment Trend Over Time (Monthly Counts)")

        tmp = df.copy()
        tmp["created_date"] = pd.to_datetime(tmp["created_date"], errors="coerce")
        tmp["sentiment"] = tmp["sentiment"].astype(str).str.title()
        tmp = tmp[tmp["sentiment"].isin(["Positive", "Negative"])]
        tmp["month_dt"] = tmp["created_date"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            tmp.groupby(["month_dt", "sentiment"])
               .size()
               .reset_index(name="count")
        )

        if not monthly.empty:
            month_range = pd.date_range(
                start=monthly["month_dt"].min(),
                end=monthly["month_dt"].max(),
                freq="MS",
            )
            all_idx = pd.MultiIndex.from_product(
                [month_range, ["Positive", "Negative"]],
                names=["month_dt", "sentiment"]
            )
            monthly = (
                monthly.set_index(["month_dt", "sentiment"])
                       .reindex(all_idx, fill_value=0)
                       .reset_index()
            )

        color_scale_trend = alt.Scale(
            domain=["Positive", "Negative"],
            range=["#2ecc71", "#e74c3c"]
        )

        trend_chart = (
            alt.Chart(monthly)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_dt:T", title="Month", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("count:Q", title="Number of Comments"),
                color=alt.Color("sentiment:N", scale=color_scale_trend, title="Sentiment"),
                detail="sentiment:N",
                tooltip=[
                    alt.Tooltip("month_dt:T", title="Month", format="%b %Y"),
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("count:Q", title="# Comments"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(trend_chart, use_container_width=True)

    # AI Summary
    st.subheader("AI Summary — Likes, Dislikes, Recommendations")
    try:
        summary = summarize_comments(
            st.session_state["comments"] or df["body"].fillna("").tolist(),
            model=OPENAI_MODEL,
            max_comments=80,
        )
        st.markdown(summary)
    except Exception as e:
        st.error(f"Error in AI summary: {e}")

    # Sample Comments (Top by Upvotes)
    st.subheader("Sample Comments (Top by Upvotes)")
    filtered_df = df[df["body"].notna() & (df["body"].str.strip() != "[deleted]")]
    cols_to_show = [c for c in ["post_title", "author", "score", "sentiment", "body"] if c in df.columns]
    st.dataframe(
        filtered_df.sort_values("score", ascending=False)[cols_to_show].head(30).reset_index(drop=True),
        use_container_width=True,hide_index=True
    )

    # Download Raw Comments as CSV (inside the main block)
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Raw Comments (CSV)",
        data=csv,
        file_name="reddit_comments.csv",
        mime="text/csv",
    )

else:
    st.info("Run an analysis to see results here.")
