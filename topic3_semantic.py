# %%
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 120)
print("✅ Libraries loaded")

# %%
raw  = pd.read_csv("Tweets.csv")[["tweet_id", "airline", "airline_sentiment", "text"]]
proc = pd.read_csv("tweets_preprocessed (1).csv")

print("Raw rows:          ", len(raw))
print("Preprocessed rows: ", len(proc))

raw["tweet_id"]  = raw["tweet_id"].astype(str)
proc["tweet_id"] = proc["tweet_id"].astype(str)

labels = raw.set_index("tweet_id")[["airline", "airline_sentiment", "text"]]
df = proc[["tweet_id", "clean_text"]].join(labels, on="tweet_id")

print("\nJoined rows:       ", len(df))
print("Missing labels:    ", df["airline_sentiment"].isna().sum())
print("Missing clean_text:", df["clean_text"].isna().sum())
print(df.head())

# %%
def analyze(text, threshold=0.1):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, "neutral"
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    if pol > threshold:
        label = "positive"
    elif pol < -threshold:
        label = "negative"
    else:
        label = "neutral"
    return pol, sub, label

for t in ["I love this airline! Best ever!",
          "Terrible service, my flight was cancelled.",
          "The flight departs at 3pm."]:
    print(t, "→", analyze(t))

# %%
print("Running TextBlob on", len(df), "tweets... (1–2 min)")
df[["polarity", "subjectivity", "tb_label"]] = df["clean_text"].apply(
    lambda t: pd.Series(analyze(t))
)
print("✅ Done")
print(df[["clean_text", "airline_sentiment", "polarity", "subjectivity", "tb_label"]].head(10))

# %%
print(f"Overall polarity:     {df['polarity'].mean():+.3f}")
print(f"Overall subjectivity: {df['subjectivity'].mean():.3f}")
print("\nTextBlob label distribution:")
print(df["tb_label"].value_counts())
print("\nGround-truth label distribution:")
print(df["airline_sentiment"].value_counts())

# %%
labels_order = ["negative", "neutral", "positive"]
print("Accuracy:", round(accuracy_score(df["airline_sentiment"], df["tb_label"]), 3))
print("\nClassification report:")
print(classification_report(df["airline_sentiment"], df["tb_label"], labels=labels_order))
cm = pd.DataFrame(
    confusion_matrix(df["airline_sentiment"], df["tb_label"], labels=labels_order),
    index=[f"true_{l}" for l in labels_order],
    columns=[f"pred_{l}" for l in labels_order],
)
print("\nConfusion matrix:")
print(cm)

# %%
airline_summary = df.groupby("airline").agg(
    n_tweets=("tweet_id", "count"),
    mean_polarity=("polarity", "mean"),
    mean_subjectivity=("subjectivity", "mean"),
    pct_negative_textblob=("tb_label", lambda x: (x == "negative").mean()),
    pct_negative_truth=("airline_sentiment", lambda x: (x == "negative").mean()),
).round(3).sort_values("mean_polarity")
print(airline_summary)

# %%
mask = (df["airline_sentiment"] == "negative") & (df["tb_label"] != "negative")
print("1) Real negatives TextBlob misread:")
print(df.loc[mask, ["clean_text", "polarity", "tb_label"]].head(10).to_string(index=False))

print("\n2) Slang check:")
for word in ["sick", "slaps", "insane", "lit"]:
    hits = df[df["clean_text"].str.contains(rf"\b{word}\b", case=False, regex=True, na=False)]
    print(f"   '{word}': {len(hits)} occurrences")
    if len(hits):
        print(hits[["clean_text", "airline_sentiment", "polarity", "tb_label"]]
              .head(2).to_string(index=False))

print("\n3) Dataset label-noise candidates:")
noise = df[(df["airline_sentiment"] == "positive") & (df["polarity"] < -0.3)]
print(f"   Count: {len(noise)}")
print(noise[["clean_text", "airline_sentiment", "polarity", "tb_label"]]
      .head(5).to_string(index=False))

# %%
order = ["negative", "neutral", "positive"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df["tb_label"].value_counts().reindex(order).plot.bar(
    ax=ax[0], color=["#e74c3c", "#f1c40f", "#27ae60"],
    title="TextBlob predicted sentiment")
df["airline_sentiment"].value_counts().reindex(order).plot.bar(
    ax=ax[1], color=["#e74c3c", "#f1c40f", "#27ae60"],
    title="Human-annotated sentiment")
plt.tight_layout()
plt.savefig("sentiment_comparison.png", dpi=120)
print("✅ Saved sentiment_comparison.png")

# %%
output_cols = ["tweet_id", "airline", "airline_sentiment",
               "clean_text", "polarity", "subjectivity", "tb_label"]
df[output_cols].to_csv("topic3_sentiment_results.csv", index=False)
print("✅ Saved topic3_sentiment_results.csv:", df[output_cols].shape)