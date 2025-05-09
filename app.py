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
        .ad-image { width: 60%; margin: auto; display: block; }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()
df["overview"].fillna("Overview not available", inplace=True)

# TF-IDF vectorization for recommendations
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"])
similarity = cosine_similarity(vector)

# Fetch movie details from TMDb
def get_movie_info_batch(movie_titles):
    results = {}
    for title in movie_titles:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        response = requests.get(url)
        data = response.json()
        if data["results"]:
            result = data["results"][0]
            results[title] = {
                "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
                "rating": result.get("vote_average", "N/A"),
                "vote_count": result.get("vote_count", "N/A")
            }
        else:
            results[title] = {"poster_url": None, "rating": "N/A", "vote_count": "N/A"}
    return results

# Fetch trending movies (For Auto-Sliding Banner)
def get_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return [{"title": movie["title"], "overview": movie["overview"], "poster_url": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None} for movie in data["results"][:4]] if data["results"] else []

# Auto-Sliding Featured Movies
trending_movies = get_trending_movies()
count = st_autorefresh(interval=5000, key="auto_refresh")  # Auto-refresh every 5s
current_movie = trending_movies[count % len(trending_movies)]
st.image(current_movie["poster_url"], use_container_width=True, caption=f"🔥 Featured: {current_movie['title']}")
st.markdown(f"📖 {current_movie['overview']}")

# Movie recommendation function with fuzzy matching
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    best_match = process.extractOne(movie_title_lower, movie_list)
    results = []

    if best_match and best_match[1] > 80:
        matched_title = best_match[0]
        idx = movie_list.index(matched_title)
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        movie_info_batch = get_movie_info_batch([searched_movie_title])

        st.subheader(f"✅ Your searched movie: {searched_movie_title}")
        st.markdown(f"📖 **Overview:** {searched_movie_overview}")
        st.markdown(f"⭐ **Rating:** {movie_info_batch[searched_movie_title]['rating']} / 10 ({movie_info_batch[searched_movie_title]['vote_count']} votes)")

        if movie_info_batch[searched_movie_title]["poster_url"]:
            st.image(movie_info_batch[searched_movie_title]["poster_url"], caption=searched_movie_title, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", use_container_width=True)

        recommended_movies = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]
        recommended_titles = [df.loc[i[0], "title"] for i in recommended_movies]
        movie_info_batch.update(get_movie_info_batch(recommended_titles))

        results = [{"title": title, "overview": df.loc[df["title"] == title, "overview"].values[0], "poster_url": movie_info_batch[title]["poster_url"]} for title in recommended_titles]

    return results

# Streamlit UI
st.title("🎬 IMDb-Style AI Movie Recommender")

# Movie Search Input
movie_input = st.text_input("🔍 Search for a movie:", "")

if st.button("Recommend"):
    if movie_input:
        results = recommend(movie_input)

        # IMDb-Style Movie Grid
        st.subheader("🔹 Recommended Movies")
        cols = st.columns(3)
        for i, movie in enumerate(results):
            with cols[i % 3]:
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(movie["poster_url"] if movie["poster_url"] else "https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption=movie["title"], use_container_width=True)
                st.markdown(f'<p class="rating">⭐ {movie["title"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="overview">{movie["overview"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<button class="button">More Like This</button>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Add User Rating Feature
if movie_input:
    rating = st.slider("⭐ Rate this movie:", 1, 10, 5)
    st.write(f"You rated {movie_input} {rating}/10!")
