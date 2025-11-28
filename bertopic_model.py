# bertopic_jira.py

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

def main():
    # 1️⃣ Load Jira tickets from input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        tickets = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(tickets)} tickets.")

    # 2️⃣ Create embeddings model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3️⃣ Create BERTopic model
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)

    # 4️⃣ Fit the model
    topics, probs = topic_model.fit_transform(tickets)

    # 5️⃣ Print top 10 topics
    print("Top 10 topics:")
    print(topic_model.get_topic_info().head(10))

    # 6️⃣ Save visualizations to HTML files (script-friendly)
    topic_model.visualize_barchart(top_n_topics=10).write_html("topic_barchart.html")
    topic_model.visualize_topics().write_html("topic_map.html")
    topic_model.visualize_heatmap(n_topics=10).write_html("topic_heatmap.html")
    print("Saved interactive visualizations as HTML files.")

    # 7️⃣ Save tickets with topics
    df = pd.DataFrame({"ticket": tickets, "topic": topics})
    df.to_csv("tickets_with_topics.csv", index=False)
    print("Saved tickets with topic assignments to tickets_with_topics.csv")

if __name__ == "__main__":
    main()
