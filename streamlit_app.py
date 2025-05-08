import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
df = pd.read_csv("movie.csv")  # your merged dataset
df.fillna('', inplace=True)

# Combine features
df["combined"] = df["GENRE"] + " " + df["ACTOR"] + " " + df["DIRECTOR"]

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie):
    movie = movie.lower().strip()
    if movie not in df["MOVIE"].str.lower().values:
        return []
    idx = df[df["MOVIE"].str.lower() == movie].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return df["MOVIE"].iloc[[i[0] for i in scores]].tolist()

# Streamlit GUI
st.title("🎥 Movie Recommendation System")

movie_input = st.text_input("Enter a movie title to get recommendations")
if st.button("Recommend"):
    if movie_input:
        recommendations = recommend(movie_input)
        if recommendations:
            st.subheader("Recommended Movies:")
            for rec in recommendations:
                st.write("🎬", rec)
        else:
            st.warning("Movie not found. Please check the title.")

