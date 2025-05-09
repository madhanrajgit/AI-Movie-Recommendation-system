import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from streamlit_autorefresh import st_autorefresh

# TMDb API Key
API_KEY = "887f725faa2dadb468b5baef8c697023"

# Apply IMDb-style dark theme
st.markdown("""
    <style>
        body { background-color: #121212; color: white; font-family: Arial, sans-serif; }
        .title { font-size: 24px; font-weight: bold; color: gold; text-align: center; }
        .rating { font-size: 18px; color: lightgreen; }
        .overview { font-size: 14px; color: white; }
        .movie-container { border: 1px solid #444; padding: 10px; border-radius: 10px; background-color: #222; text-align: center; }
        .button { background-color: gold; color: black; font-size: 16px; padding: 5px; border-radius: 8px; }
        .ad-image { width: 30%; margin: auto; display: block; }
    </style>
""", unsafe_allow_html=True)

# Load dataset safely
try:
    df = pd.read_csv("merged_movies.csv")
    if df.empty:
        raise pd.errors.EmptyDataError  # If no data is present, raise an error
    df.columns = df.columns.str.strip()
    df["overview"].fillna("Overview not available", inplace=True)
except (FileNotFoundError, pd.errors.EmptyDataError):
    st.error("⚠️ Error: The dataset `merged_movies.csv` is missing or empty. Please check the file and try again.")
    df = pd.DataFrame()  # Initialize an empty DataFrame to avoid crashes

# TF-IDF vectorization for recommendations (only if data exists)
if not df.empty:
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)

# Ensure recommendations persist
if "recommended_movies" not in st.session_state:
    st.session_state.recommended_movies = []

# Fetch movie details from TMDb
def get_movie_info(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data["results"]:
        result = data["results"][0]
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
            "rating": result.get("vote_average", "N/A"),
            "vote_count": result.get("vote_count", "N/A"),
            "id": result["id"]
        }
    return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "id": None}

# Genre-Based Recommendations
def get_genre_recommendations(movie_title):
    if df.empty:
        return []
    genres = df[df["title"].str.lower() == movie_title.lower()]["genres"]
    if genres.empty:
        return []
    genre_movies = df[df["genres"].str.contains(genres.values[0], na=False)]
    return genre_movies["title"].tolist()

# Overview-Based Recommendations
def get_overview_recommendations(movie_title):
    if df.empty:
        return []
    movie_index = df[df["title"].str.lower() == movie_title.lower()].index
    if movie_index.empty:
        return []
    scores = list(enumerate(similarity[movie_index[0]]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]]["title"] for i in scores]

# Crew-Based Recommendations (Directors, Writers, Actors)
def get_crew_recommendations(movie_title):
    if df.empty:
        return []
    crew_members = df[df["title"].str.lower() == movie_title.lower()]["crew"]
    if crew_members.empty:
        return []
    crew_movies = df[df["crew"].str.contains(crew_members.values[0], na=False)]
    return crew_movies["title"].tolist()

# Generate Movie Details Page
def show_movie_details(movie_title):
    movie_info = get_movie_info(movie_title)
    st.subheader(f"🎬 {movie_title}")
    
    if movie_info["poster_url"]:
        st.image(movie_info["poster_url"], caption=movie_title, use_container_width=True)
    
    st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")
    
    genre_recommendations = get_genre_recommendations(movie_title)
    overview_recommendations = get_overview_recommendations(movie_title)
    crew_recommendations = get_crew_recommendations(movie_title)
    
    st.subheader("🎯 Recommended Movies Based on Genre")
    for movie in genre_recommendations:
        st.write(f"🔹 [{movie}](?selected_movie={movie})")
    
    st.subheader("🎯 Recommended Movies Based on Overview")
    for movie in overview_recommendations:
        st.write(f"🔹 [{movie}](?selected_movie={movie})")
    
    st.subheader("🎯 Recommended Movies Based on Crew")
    for movie in crew_recommendations:
        st.write(f"🔹 [{movie}](?selected_movie={movie})")

# UI Layout
st.title("🎬 IMDb-Style AI Movie Recommender")
col1, col2 = st.columns([4, 1])

with col1:
    movie_input = st.text_input("🔍 Search for a movie:", "")

with col2:
    if st.button("🔍 Search") and movie_input:
        st.session_state.selected_movie = movie_input

# Display Movie Details if Selected
if "selected_movie" in st.session_state:
    show_movie_details(st.session_state.selected_movie)

# Auto-Sliding Featured Movies
if not movie_input or st.button("🏠 Back to Home"):
    trending_movies = [get_movie_info(title) for title in df.sample(4)["title"].tolist()]
    for movie in trending_movies:
        st.image(movie["poster_url"], use_container_width=True, caption=f"🔥 Featured: {movie['id']}")
