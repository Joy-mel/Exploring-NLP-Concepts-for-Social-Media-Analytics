"""
Topic 2: Syntactic Analysis of Social Media Text
POS tagging + dependency parsing with spaCy.

Usage:
    python syntactic_analysis.py                      # uses the built-in sample tweets
    python syntactic_analysis.py tweets.csv text      # uses a CSV (file, text-column name)
                                                      # e.g. the dataset from Topic 1

Setup (once):
    pip install spacy pandas matplotlib
    python -m spacy download en_core_web_sm
"""
import html
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy import displacy

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. DATA
SAMPLE_TWEETS = [
    "Just landed in Nairobi and the traffic is already crazy 😂 #KenyaTwitter",
    "The new phone camera is amazing, but the battery dies too fast.",
    "@SafaricomPLC why is my internet so slow today??",
    "Can't believe we lost the match again, our defence was terrible!",
    "u r gonna luv this new song, trust me lol",
    "JKUAT students are building apps that solve real problems in Kenya.",
    "Loving the weather today, going for a long walk with friends 🌞",
    "This government keeps ignoring the youth and nobody is talking about it.",
    "Great service at the new cafe in Juja, the staff were very friendly.",
    "omg the exam was sooo hard, i think i failed",
    "Breaking: heavy rain floods several roads in Nairobi https://t.co/xyz",
    "She sold her old laptop and bought a better one yesterday.",
    "Not happy with the delivery, my order arrived late and damaged.",
    "Data science is hard but the projects are worth every minute!",
    "Thanks @KenyaPower for restoring electricity so quickly 🙏",
]

def load_posts(csv_path=None, column="text", limit=200):
    if csv_path:
        df = pd.read_csv(csv_path)
        texts = df[column].dropna().astype(str)
        # random sample (fixed seed = repeatable): files are often sorted, e.g. by airline
        if len(texts) > limit:
            texts = texts.sample(limit, random_state=42)
        return texts.tolist()
    return SAMPLE_TWEETS

# ------------------------------------------------- 2. LIGHT CLEANING ONLY
# Syntax needs word order and function words, so we do NOT remove stop words,
# stem, or lower-case here (unlike Topic 1 style preprocessing).
URL_RE = re.compile(r"https?://\S+")

def light_clean(text):
    text = html.unescape(text)            # "&amp;" -> "&"
    text = URL_RE.sub("URL", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------- 3. MODEL
nlp = spacy.load("en_core_web_sm")

def analyse(posts):
    pos_rows, dep_rows, docs = [], [], []
    for i, post in enumerate(posts, start=1):
        doc = nlp(light_clean(post))
        docs.append(doc)
        for tok in doc:
            if tok.is_space:
                continue
            pos_rows.append({
                "tweet_id": i, "token": tok.text,
                "pos_universal": tok.pos_,   # coarse tag (NOUN, VERB ...)
                "pos_fine": tok.tag_,        # Penn Treebank tag (NN, VBZ ...)
                "lemma": tok.lemma_,
            })
            dep_rows.append({
                "tweet_id": i, "token": tok.text,
                "dependency": tok.dep_,      # relation to its head
                "head": tok.head.text,
                "head_pos": tok.head.pos_,
                "children": " ".join(c.text for c in tok.children),
            })
    return docs, pd.DataFrame(pos_rows), pd.DataFrame(dep_rows)

# ------------------------------------------------------------ 4. OUTPUTS
def pos_chart(pos_df):
    counts = pos_df["pos_universal"].value_counts()
    ax = counts.plot(kind="bar", figsize=(9, 4), color="#2f6f9f")
    ax.set_title("Part-of-speech distribution across the tweets")
    ax.set_xlabel("Universal POS tag")
    ax.set_ylabel("Number of tokens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "pos_distribution.png", dpi=200)
    plt.close()

def save_tree(doc, name):
    svg = displacy.render(doc, style="dep", jupyter=False,
                          options={"compact": False, "distance": 110,
                                   "bg": "#ffffff", "color": "#1a1a1a"})
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")

def noun_chunks_and_roots(docs):
    rows = []
    for i, doc in enumerate(docs, start=1):
        for sent in doc.sents:
            rows.append({
                "tweet_id": i,
                "root_verb": sent.root.text,
                "subjects": ", ".join(t.text for t in sent if t.dep_ in ("nsubj", "nsubjpass")),
                "objects": ", ".join(t.text for t in sent if t.dep_ in ("dobj", "obj", "pobj")),
                "noun_phrases": " | ".join(c.text for c in sent.noun_chunks),
            })
    return pd.DataFrame(rows)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    column = sys.argv[2] if len(sys.argv) > 2 else "text"
    posts = load_posts(csv_path, column)
    print(f"Analysing {len(posts)} posts\n")

    docs, pos_df, dep_df = analyse(posts)
    struct_df = noun_chunks_and_roots(docs)

    pos_df.to_csv(OUT / "pos_tags.csv", index=False)
    dep_df.to_csv(OUT / "dependencies.csv", index=False)
    struct_df.to_csv(OUT / "sentence_structure.csv", index=False)
    pos_chart(pos_df)

    # --- Console output
    print("=== POS TAGGING (first tweet) ===")
    print(pos_df[pos_df.tweet_id == 1].to_string(index=False))
    print("\n=== DEPENDENCY PARSING (first tweet) ===")
    print(dep_df[dep_df.tweet_id == 1].to_string(index=False))
    print("\n=== SENTENCE STRUCTURE (subject / verb / object) ===")
    print(struct_df.head(8).to_string(index=False))
    print("\n=== MOST COMMON POS TAGS ===")
    print(Counter(pos_df["pos_universal"]).most_common(8))

    # --- Dependency diagrams: one clean tweet, one noisy tweet
    if csv_path is None:
        clean_idx, noisy_idx = 1, 4   # "The new phone camera ..." / "u r gonna luv ..."
    else:
        # pick the first two medium-length tweets (8-14 tokens) so the diagrams stay readable
        medium = [i for i, d in enumerate(docs) if 8 <= len(d) <= 14] or list(range(len(docs)))
        clean_idx, noisy_idx = medium[0], medium[min(1, len(medium) - 1)]
        print(f"\nDiagram tweets: #{clean_idx + 1} and #{noisy_idx + 1} (see dep_tree_*.svg)")
    save_tree(docs[clean_idx], "dep_tree_clean")
    save_tree(docs[noisy_idx], "dep_tree_noisy")
    # Shorter clauses give larger, easier-to-read diagrams for the report
    save_tree(nlp("The new phone camera is amazing"), "dep_tree_clause")
    save_tree(nlp("u r gonna luv this new song"), "dep_tree_noisy_clause")

    # --- Where does the parser struggle? Compare informal vs standard wording
    print("\n=== CHALLENGE DEMO: informal vs standard wording ===")
    pairs = [("u r gonna luv this new song", "You are going to love this new song"),
             ("omg the exam was sooo hard", "Oh my god, the exam was so hard")]
    for informal, formal in pairs:
        for label, s in (("informal", informal), ("standard", formal)):
            d = nlp(s)
            print(f"[{label}] " + " ".join(f"{t.text}/{t.pos_}" for t in d))
        print()

if __name__ == "__main__":
    main()
