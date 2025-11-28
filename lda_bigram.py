import pandas as pd
import string
import spacy
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser

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
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize & lemmatize
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.is_alpha]
    return tokens

# --------------------------------------------------------
# MAIN ENTRY POINT (IMPORTANT FOR macOS)
# --------------------------------------------------------
if __name__ == "__main__":
    # Load input file (each issue title is one line)
    stop_words = set(stopwords.words('english'))
    stop_words.add('I')
    stop_words.add('want')
    stop_words.add('I want')

    with open("input.txt", "r", encoding="utf8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]

    # Clean all issue titles
    data_words = [preprocess(doc) for doc in documents]


    # -----------------------------
    # Build bigram model
    # -----------------------------
    bigram = Phrases(data_words, min_count=2, threshold=5)  # tune these values
    bigram_mod = Phraser(bigram)
    data_words_bigrams = [bigram_mod[doc] for doc in data_words]

    # -----------------------------
    # Create dictionary and corpus
    # -----------------------------
    id2word = corpora.Dictionary(data_words_bigrams)
    corpus = [id2word.doc2bow(text) for text in data_words_bigrams]

    # Train LDA Model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=5,       # adjust number of topics
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # -----------------------------
    # Print topics
    # -----------------------------
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    # -----------------------------
    # Visualize topics
    # -----------------------------
    #pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, "topics.html")
