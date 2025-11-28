import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def main():
    # Read your Jira headings
    with open("input.txt", "r") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    # Create BERTopic model
    topic_model = BERTopic(
        vectorizer_model=CountVectorizer(ngram_range=(1, 2))  # include bigrams
    )

    topics, probs = topic_model.fit_transform(docs)

    # Optional: reduce number of topics
    topic_model.reduce_topics(docs, nr_topics=50)

    # Visualize
    fig1 = topic_model.visualize_topics()
    fig1.write_html("topic_overview.html")

    fig2 = topic_model.visualize_heatmap()  # NO n_topics argument
    fig2.write_html("topic_heatmap.html")

    # Top 10 topics
    print(topic_model.get_topic_info().head(10))

if __name__ == "__main__":
    main()
