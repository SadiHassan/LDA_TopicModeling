#!/usr/bin/env python3
import re
import spacy
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pandas as pd

# ---------- INITIAL SETUP ----------
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_md")  # or en_core_web_sm if small

# ---------- HELPER FUNCTIONS ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return tokens

def extract_bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def extract_entities(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]

# ---------- MAIN ----------
def main():
    # 1. Read Jira headings
    with open("input.txt", "r") as f:
        headings = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(headings)} Jira headings.")

    # 2. Extract tokens, bigrams, entities
    all_tokens = []
    all_bigrams = []
    all_entities = []
    for h in headings:
        tokens = preprocess_text(h)
        all_tokens.extend(tokens)
        all_bigrams.extend(extract_bigrams(tokens))
        all_entities.extend(extract_entities(h))

    # 3. Frequency charts
    top_tokens = Counter(all_tokens).most_common(20)
    top_bigrams = Counter(all_bigrams).most_common(20)
    top_entities = Counter(all_entities).most_common(20)

    print("\nTop 20 unigrams:", top_tokens)
    print("\nTop 20 bigrams:", top_bigrams)
    print("\nTop 20 entities:", top_entities)

    # Plot unigram frequencies
    tokens_df = pd.DataFrame(top_tokens, columns=["Token", "Count"])
    plt.figure(figsize=(10,6))
    plt.bar(tokens_df["Token"], tokens_df["Count"])
    plt.xticks(rotation=45)
    plt.title("Top 20 Unigrams")
    plt.tight_layout()
    plt.savefig("top_unigrams.png")
    plt.close()

    # Plot bigram frequencies
    bigrams_df = pd.DataFrame([" ".join(b) for b, c in top_bigrams], columns=["Bigram"])
    bigrams_df["Count"] = [c for b, c in top_bigrams]
    plt.figure(figsize=(10,6))
    plt.bar(bigrams_df["Bigram"], bigrams_df["Count"])
    plt.xticks(rotation=45)
    plt.title("Top 20 Bigrams")
    plt.tight_layout()
    plt.savefig("top_bigrams.png")
    plt.close()

    # 4. Co-occurrence graph
    G = nx.Graph()
    # Add nodes
    for token, _ in top_tokens:
        G.add_node(token)
    # Add edges for bigrams
    for (w1, w2), count in top_bigrams:
        if w1 in G.nodes and w2 in G.nodes:
            G.add_edge(w1, w2, weight=count)

    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G)
    net.show("cooccurrence_graph.html")

    # 5. Clustering with embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(headings, show_progress_bar=True)
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Scatter plot
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(
        umap_embeddings[:,0],
        umap_embeddings[:,1],
        c=cluster_labels,
        cmap='tab20',
        s=50
    )
    plt.colorbar(scatter)
    plt.title("Jira Heading Clusters")
    plt.tight_layout()
    plt.savefig("jira_clusters.png")
    plt.close()

    # Save heading + cluster info
    df = pd.DataFrame({"Heading": headings, "Cluster": cluster_labels})
    df.to_csv("jira_headings_clusters.csv", index=False)

    print("Analysis completed! Output files generated:")
    print(" - top_unigrams.png")
    print(" - top_bigrams.png")
    print(" - cooccurrence_graph.html")
    print(" - jira_clusters.png")
    print(" - jira_headings_clusters.csv")

if __name__ == "__main__":
    main()
