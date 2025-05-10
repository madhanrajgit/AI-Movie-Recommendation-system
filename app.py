import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO

# TMDb API Key (Replace with your actual API key)
API_KEY = "887f725faa2dadb468b5baef8c697023"

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()
df["overview"].fillna("Overview not available", inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"].fillna(""))
similarity = cosine_similarity(vector)

# TMDb: Fetch movie data (poster, rating)
def get_movie_data(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
    response = requests.get(url)
    data = response.json()
    if data["results"]:
        result = data["results"][0]
        poster_path = result.get("poster_path")
        rating = result.get("vote_average", 0)
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return poster_url, rating
    return None, 0.0

# Recommend function
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    results = []

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        poster_url, rating = get_movie_data(searched_movie_title)

        st.markdown("""
            <div style='border: 1px solid #444; padding: 20px; border-radius: 10px; background-color: #222;'>
                <h3 style='color: gold;'>✅ {}</h3>
                <p style='color: white;'><b>📖 Overview:</b> {}</p>
                <p style='color: gold;'>⭐ Rating: {}/10</p>
            </div>
        """.format(searched_movie_title, searched_movie_overview, round(rating, 1)), unsafe_allow_html=True)

        if poster_url:
            st.image(poster_url, width=200)

        recommended_movies = sorted(
            list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
        )[1:6]

        for i in recommended_movies:
            title = df.loc[i[0], "title"]
            overview = df.loc[i[0], "overview"]
            poster_url, rating = get_movie_data(title)

            st.markdown("""
                <div style='margin-top: 20px; padding: 15px; border-radius: 10px; background-color: #111;'>
                    <h4 style='color: gold;'>🎬 {}</h4>
                    <p style='color: white;'><b>📖 Overview:</b> {}</p>
                    <p style='color: gold;'>⭐ Rating: {}/10</p>
                </div>
            """.format(title, overview, round(rating, 1)), unsafe_allow_html=True)

            if poster_url:
                st.image(poster_url, width=150)
    else:
        st.error(f"❌ Movie '{movie_title}' not found. Showing top popular movies!")
        top_popular = df.sort_values("popularity", ascending=False).head(5)

        for _, row in top_popular.iterrows():
            title = row["title"]
            overview = row["overview"]
            poster_url, rating = get_movie_data(title)

            st.markdown("""
                <div style='margin-top: 20px; padding: 15px; border-radius: 10px; background-color: #111;'>
                    <h4 style='color: gold;'>🔥 {}</h4>
                    <p style='color: white;'><b>📖 Overview:</b> {}</p>
                    <p style='color: gold;'>⭐ Rating: {}/10</p>
                </div>
            """.format(title, overview, round(rating, 1)), unsafe_allow_html=True)

            if poster_url:
                st.image(poster_url, width=150)

# App layout and design
st.set_page_config(page_title="IMDb Style Movie Recommender", layout="centered", page_icon="🎥")
st.markdown("""
    <style>
        body {
            background-color: #000;
        }
        .reportview-container {
            background-color: #111;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #111;
        }
        h1, h2, h3, h4 {
            color: gold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🍿 AI Movie Recommender System")
st.markdown("Enter a movie name to get recommendations based on its <b>overview</b> and get <b>ratings</b> from TMDb.", unsafe_allow_html=True)

movie_input = st.text_input("🎬 Enter a movie name:", "")

if st.button("🎯 Recommend"):
    if movie_input:
        recommend(movie_input)
