# Social Media Sentiment Analysis

This project analyzes the Prodigy InfoTech Task 4 Twitter entity sentiment
dataset. It summarizes the supplied sentiment labels and compares sentiment
patterns across topics/entities.

## Dataset

Download `twitter_training.csv` from the
[Task 4 dataset folder](https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204)
and place it beside `analyze_sentiment.py`. The CSV has no header row; its
columns are tweet ID, entity/topic, sentiment, and tweet text. The dataset
originates from the
[Twitter Entity Sentiment Analysis dataset on Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## Run

Install the requirements and run the analysis:

```powershell
python -m pip install -r requirements.txt
python task4.py
```

The entrypoint reads `twitter_training.csv` by default and writes charts and CSV
summaries to `output\`. Custom paths and the number of entities in the chart
can be supplied:

```powershell
python task4.py --data path\to\twitter_training.csv --output results --top-n 20
```

## Outputs

- `sentiment_distribution.png` and `sentiment_summary.csv`: counts and shares
  for all available labels, including `Irrelevant`.
- `sentiment_by_entity.png` and `entity_sentiment_summary.csv`: polarity shares
  for the most represented entities.
- `top_terms_by_sentiment.png` and `top_terms_by_sentiment.csv`: frequent
  non-stopword terms in positive, neutral, and negative tweets.

`Irrelevant` rows remain in the overall distribution but are excluded from
polarity comparisons and term summaries. Topic comparisons use the proportion
of positive, neutral, and negative tweets within each entity, so entities with
different tweet volumes can be compared. The supplied sentiment labels are
analyzed as-is; this project does not independently classify tweet text.
