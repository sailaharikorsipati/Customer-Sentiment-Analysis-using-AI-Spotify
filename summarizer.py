from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_comments(comments, model="gpt-3.5-turbo", max_comments=80):
    """
    Summarize a list of comments into likes, dislikes, and recommendations.
    Optional parameters:
        model: OpenAI model to use (default: gpt-3.5-turbo)
        max_comments: maximum number of comments to include in the summary
    """
    if not comments:
        return "No comments to summarize."

    # Limit the number of comments if needed
    comments_to_summarize = comments[:max_comments]

    prompt = (
        "Summarize customer feedback into three sections:\n\n"
        "Likes:\nDislikes:\nRecommendations:\n\nComments:\n"
        + "\n".join(comments_to_summarize)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ("You are a senior marketing strategist and consumer insights expert. "
                "Your job is to analyze customer feedback and translate it into clear business insights. "
                "Always structure your output into three sections: Likes, Dislikes, and Recommendations. "
                "Write in concise bullet points, 3 for each section, focusing on themes that would matter to product managers "
                "and executives making strategic decisions.")},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI API error: {e}"
