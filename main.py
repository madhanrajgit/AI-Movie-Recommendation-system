import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from streamlit_autorefresh import st_autorefresh

# TMDb API Key
API_KEY = "887f725faa2dadb468b5baef8c697023"

# Load dataset safely
try:
    df = pd.read_csv("merged_movies.csv")
    if df.empty:
        raise pd.errors.EmptyDataError  # If no data is present, raise an error
    df.columns = df.columns.str.strip()
    df.fillna("Information not available", inplace=True)
except (FileNotFoundError, pd.errors.EmptyDataError):
    st.error("⚠️ Error: The dataset `merged_movies.csv` is missing or empty.")
    df = pd.DataFrame()  # Initialize an empty DataFrame

# TF-IDF vectorization for recommendations (only if data exists)
if not df.empty:
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)

# Function to fetch movie details from TMDb
def get_movie_info(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url).json()
    if response["results"]:
        result = response["results"][0]
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
            "rating": result.get("vote_average", "N/A"),
            "vote_count": result.get("vote_count", "N/A"),
            "id": result["id"]
        }
    return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "id": None}

# Function to generate recommendations
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    best_match = process.extractOne(movie_title_lower, movie_list)

    if not best_match or best_match[1] < 80:
        st.warning(f"⚠️ Movie '{movie_title}' not found. Try another title!")
        return []

    matched_title = best_match[0]
    idx = movie_list.index(matched_title)
    
    st.session_state.selected_movie = matched_title
    st.session_state.recommended_movies = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]

# Function to show movie details page
def show_movie_details(movie_title):
    movie_info = get_movie_info(movie_title)
    st.subheader(f"🎬 {movie_title}")

    if movie_info["poster_url"]:
        st.image(movie_info["poster_url"], caption=movie_title, use_container_width=True)

    st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")

    # Fetch recommendations based on genre, overview, and crew
    st.subheader("🔹 Recommended Movies Based on Genre")
    for movie in df[df["genres"].str.contains(df.loc[df["title"] == movie_title, "genres"].values[0], na=False)]["title"].tolist():
        st.write(f"🔹 [{movie}](?selected_movie={movie})")

    st.subheader("🔹 Recommended Movies Based on Overview")
    overview_idx = df[df["title"] == movie_title].index[0]
    overview_recommendations = sorted(list(enumerate(similarity[overview_idx])), key=lambda x: x[1], reverse=True)[1:6]
    for movie in [df.iloc[i[0]]["title"] for i in overview_recommendations]:
        st.write(f"🔹 [{movie}](?selected_movie={movie})")

    st.subheader("🔹 Recommended Movies Based on Crew")
    crew_members = df.loc[df["title"] == movie_title, "crew"].values[0]
    for movie in df[df["crew"].str.contains(crew_members, na=False)]["title"].tolist():
        st.write(f"🔹 [{movie}](?selected_movie={movie})")

# UI Layout
st.title("🎬 IMDb-Style AI Movie Recommender")
col1, col2 = st.columns([4, 1])

with col1:
    movie_input = st.text_input("🔍 Search for a movie:", "")

with col2:
    if st.button("🔍 Search") and movie_input:
        recommend(movie_input)

# Display Movie Details if Selected
if "selected_movie" in st.session_state:
    show_movie_details(st.session_state.selected_movie)

# Auto-Sliding Featured Movies
if not movie_input or st.button("🏠 Back to Home"):
    trending_movies = [get_movie_info(title) for title in df.sample(4)["title"].tolist()]
    for movie in trending_movies:
        st.image(movie["poster_url"], use_container_width=True, caption=f"🔥 Featured: {movie['id']}")
