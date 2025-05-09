import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()

# TF-IDF for recommendations
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"].fillna(""))
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)
        recommended_movies = sorted(
            list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
        )[1:6]

        results = [
            {"title": df.loc[i[0], "title"], "overview": df.loc[i[0], "overview"]}
            for i in recommended_movies
        ]
    else:
        top_popular = df.sort_values("popularity", ascending=False).head(5)
        results = [
            {"title": row["title"], "overview": row["overview"]}
            for _, row in top_popular.iterrows()
        ]

    return results

# Streamlit UI
st.title("🎬 AI Movie Recommender")

movie_input = st.text_input("Enter a movie name:", "")

if st.button("Recommend"):
    if movie_input:
        results = recommend(movie_input)

        if movie_input.lower() in df["title"].str.lower().tolist():
            st.subheader(f"✅ Your searched movie: {movie_input}")
        else:
            st.subheader(f"❌ Movie '{movie_input}' not found. Showing top popular movies!")

        for movie in results:
            st.write(f"**👉 {movie['title']}**")
            st.write(f"📖 {movie['overview']}\n")

