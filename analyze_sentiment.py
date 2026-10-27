import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_COLUMNS = ["tweet_id", "entity", "sentiment", "text"]
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive", "Irrelevant"]
POLARITY_ORDER = ["Negative", "Neutral", "Positive"]
STOP_WORDS = {
    "about", "after", "again", "against", "all", "also", "and", "are", "because",
    "been", "before", "being", "but", "can", "could", "did", "does", "doing",
    "com", "dont", "each", "for", "from", "get", "got", "had", "has", "have", "her",
    "here", "him", "his", "how", "into", "its", "just", "like", "lol", "make",
    "https", "more", "most", "not", "now", "off", "one", "our", "out", "over",
    "pic", "really", "she", "should", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "too", "very", "was", "way",
    "twitter", "unk", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        header=None,
        names=DATA_COLUMNS,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    data["entity"] = data["entity"].astype("string").str.strip()
    data["sentiment"] = (
        data["sentiment"].astype("string").str.strip().str.lower()
        .map({
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
            "irrelevant": "Irrelevant",
        })
        .fillna(data["sentiment"].astype("string").str.strip())
    )
    data["text"] = data["text"].astype("string").str.strip()
    valid = data["entity"].notna() & data["sentiment"].notna() & data["text"].notna()
    valid &= data["entity"].ne("") & data["text"].ne("")
    cleaned = data.loc[valid].copy()

    unexpected = sorted(set(cleaned["sentiment"].dropna()) - set(SENTIMENT_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels in dataset: {unexpected}")
    if cleaned.empty:
        raise ValueError(f"No usable rows found in {path}")
    return cleaned


def sentiment_summary(data: pd.DataFrame) -> pd.DataFrame:
    counts = data["sentiment"].value_counts()
    order = [label for label in SENTIMENT_ORDER if label in counts.index]
    summary = counts.reindex(order).rename("count").to_frame()
    summary["percentage"] = summary["count"] / len(data) * 100
    summary.index.name = "sentiment"
    return summary.reset_index()


def entity_summary(data: pd.DataFrame) -> pd.DataFrame:
    relevant = data[data["sentiment"].isin(POLARITY_ORDER)]
    grouped = (
        relevant.groupby(["entity", "sentiment"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=POLARITY_ORDER, fill_value=0)
    )
    grouped["total_relevant"] = grouped.sum(axis=1)
    grouped["positive_percentage"] = (
        grouped["Positive"] / grouped["total_relevant"] * 100
    )
    grouped["negative_percentage"] = (
        grouped["Negative"] / grouped["total_relevant"] * 100
    )
    return grouped.sort_values("total_relevant", ascending=False)


def top_terms(data: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    rows = []
    for sentiment in POLARITY_ORDER:
        texts = data.loc[data["sentiment"] == sentiment, "text"].dropna()
        counts = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
            counts.update(
                token for token in tokens
                if len(token) > 2 and token not in STOP_WORDS
            )
        rows.extend(
            {"sentiment": sentiment, "term": term, "count": count}
            for term, count in counts.most_common(limit)
        )
    return pd.DataFrame(rows, columns=["sentiment", "term", "count"])


def save_charts(
    summary: pd.DataFrame,
    by_entity: pd.DataFrame,
    terms: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    colors = {
        "Negative": "#e76f51",
        "Neutral": "#9aa0a6",
        "Positive": "#2a9d8f",
        "Irrelevant": "#8e79c6",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary["sentiment"],
        summary["percentage"],
        color=[colors[label] for label in summary["sentiment"]],
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(title="Sentiment distribution", ylabel="Tweets (%)", xlabel="")
    ax.set_ylim(0, max(summary["percentage"].max() * 1.15, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "sentiment_distribution.png", dpi=160)
    plt.close(fig)

    selected = by_entity.head(top_n).sort_values("total_relevant")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(selected))))
    left = pd.Series(0.0, index=selected.index)
    for sentiment in POLARITY_ORDER:
        values = selected[sentiment] / selected["total_relevant"] * 100
        ax.barh(
            selected.index,
            values,
            left=left,
            label=sentiment,
            color=colors[sentiment],
        )
        left += values
    ax.set(
        title=f"Sentiment by topic/entity (top {len(selected)} by relevant tweet count)",
        xlabel="Share of relevant tweets (%)",
        ylabel="Topic/entity",
        xlim=(0, 100),
    )
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "sentiment_by_entity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(POLARITY_ORDER), figsize=(15, 5))
    for ax, sentiment in zip(axes, POLARITY_ORDER):
        subset = terms[terms["sentiment"] == sentiment].sort_values("count")
        ax.barh(subset["term"], subset["count"], color=colors[sentiment])
        ax.set(title=sentiment, xlabel="Tweet count")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Most frequent terms in relevant tweets")
    fig.tight_layout()
    fig.savefig(output_dir / "top_terms_by_sentiment.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize sentiment in the Prodigy Twitter dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("twitter_training.csv"),
        help="Path to the headerless training CSV (default: twitter_training.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for charts and summary CSV files (default: output)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of entities shown in the comparison chart (default: 15)",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    with args.data.open(encoding="utf-8") as source_file:
        raw_rows = sum(1 for _ in source_file)
    data = load_dataset(args.data)
    summary = sentiment_summary(data)
    by_entity = entity_summary(data)
    terms = top_terms(data)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "sentiment_summary.csv", index=False)
    by_entity.to_csv(args.output / "entity_sentiment_summary.csv")
    terms.to_csv(args.output / "top_terms_by_sentiment.csv", index=False)
    save_charts(summary, by_entity, terms, args.output, args.top_n)

    print(f"Rows in source: {raw_rows:,}")
    print(f"Usable rows analyzed: {len(data):,}")
    print(f"Rows excluded due to missing entity, sentiment, or text: "
          f"{raw_rows - len(data):,}")
    print("\nSentiment distribution:")
    print(summary.to_string(index=False, formatters={"percentage": "{:.2f}%".format}))
    relevant_count = int(data["sentiment"].isin(POLARITY_ORDER).sum())
    print(f"\nIrrelevant tweets excluded from topic/term polarity views: "
          f"{len(data) - relevant_count:,}")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
