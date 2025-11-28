import pandas as pd
import string
import spacy
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# --------------------------------------------------------
# DOWNLOAD RESOURCES (only runs once)
# --------------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_md")

# --------------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------------
def clean_text(text):
    doc = nlp(text.lower())

    stops = set(stopwords.words("english"))

    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha               # keep only words
        and token.lemma_ not in stops   # remove stopwords
        and len(token.lemma_) > 2       # remove short words
    ]
    return tokens

# --------------------------------------------------------
# MAIN ENTRY POINT (IMPORTANT FOR macOS)
# --------------------------------------------------------
if __name__ == "__main__":
    # Load input file (each issue title is one line)
    with open("input.txt", "r", encoding="utf8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Clean all issue titles
    texts = [clean_text(line) for line in lines]

    # Create dictionary + corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA Model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        passes=30,
        random_state=42
    )

    # Print discovered topics
    print("\n===== Topics Found =====")
    topics = lda_model.print_topics(num_words=10)
    for t in topics:
        print(t)

    # Create HTML visualization
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, "topics.html")
    print("\nHTML visualization saved to topics.html")
