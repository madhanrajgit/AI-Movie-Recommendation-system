import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# --- Configuration ---
try:
    API_KEY = st.secrets[887f725faa2dadb468b5baef8c697023]  # Use Streamlit secrets for secure API key storage
except KeyError:
    API_KEY = os.getenv(887f725faa2dadb468b5baef8c697023)  # Fallback to environment variable
    if not API_KEY:
        st.error("API key not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Load and preprocess the movie dataset."""
    if not os.path.exists("merged_movies.csv"):
        st.error("Dataset 'merged_movies.csv' not found!")
        st.stop()
    
    df = pd.read_csv("merged_movies.csv")
    df.columns = df.columns.str.strip()
    required_columns = ["title", "overview", "popularity"]
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset missing required columns: title, overview, popularity")
        st.stop()
    
    df["overview"] = df["overview"].fillna("Overview not available")
    return df

# --- TF-IDF and Similarity Computation ---
@st.cache_resource
def compute_similarity(df):
    """Compute TF-IDF vectors and cosine similarity matrix."""
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)
    return similarity

# --- TMDb API Functions ---
@st.cache_data
def get_movie_info(movie_title):
    """Fetch movie details from TMDb API."""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            result = data["results"][0]
            return {
                "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
                "rating": result.get("vote_average", "N/A"),
                "vote_count": result.get("vote_count", "N/A")
            }
        return {"poster_url": None, "rating": "N/A", "vote_count": "N/A"}
    except requests.RequestException as e:
        st.error(f"Error fetching movie info for '{movie_title}': {e}")
        return {"poster_url": None, "rating": "N/A", "vote_count": "N/A"}

@st.cache_data
def get_top_rated_movies():
    """Fetch top-rated movies from TMDb API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("results", [])[:5]
    except requests.RequestException as e:
        st.error(f"Error fetching top-rated movies: {e}")
        return []

# --- Recommendation Logic ---
def recommend(movie_title, df, similarity):
    """Generate movie recommendations based on input title."""
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    results = []

    # Fuzzy matching for better search
    match = process.extractOne(movie_title_lower, movie_list)
    if match and match[1] > 80:  # Threshold for similarity
        idx = movie_list.index(match[0])
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        movie_info = get_movie_info(searched_movie_title)

        # Display searched movie
        st.subheader(f"✅ Your searched movie: {searched_movie_title}")
        st.markdown(f"📖 **Overview:** {searched_movie_overview}")
        st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")
        if movie_info["poster_url"]:
            st.image(movie_info["poster_url"], caption=searched_movie_title, width=200)
        else:
            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=200)

        # Get recommendations
        recommended_movies = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]
        results = [
            {"title": df.loc[i[0], "title"], "overview": df.loc[i[0], "overview"]}
            for i in recommended_movies
        ]
    else:
        st.subheader(f"❌ Movie '{movie_title}' not found. Showing top popular movies!")
        top_popular = df.sort_values("popularity", ascending=False).head(5)
        results = [
            {"title": row["title"], "overview": row["overview"]}
            for _, row in top_popular.iterrows()
        ]

    return results

# --- Streamlit UI ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🎬 AI Movie Recommender")
    st.markdown("Enter a movie title to get recommendations or view top-rated movies from TMDb!")

    # Load data and compute similarity
    df = load_data()
    similarity = compute_similarity(df)

    # Movie input and recommendation
    movie_input = st.text_input("Enter a movie name:", "")
    if st.button("Recommend"):
        if movie_input.strip():
            results = recommend(movie_input, df, similarity)
            
            # Display recommendations in a grid
            st.subheader("🎥 Recommended Movies")
            cols = st.columns(3)
            for i, movie in enumerate(results):
                with cols[i % 3]:
                    st.write(f"**👉 {movie['title']}**")
                    st.markdown(f"📖 **Overview:** {movie['overview']}")
                    movie_info = get_movie_info(movie['title'])
                    st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")
                    if movie_info["poster_url"]:
                        st.image(movie_info["poster_url"], caption=movie['title'], width=150)
                    else:
                        st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)
        else:
            st.warning("Please enter a movie title!")

    # Top-rated movies
    if st.button("Top Rated Movies"):
        top_movies = get_top_rated_movies()
        if top_movies:
            st.subheader("🎯 Top Rated Movies from TMDb")
            cols = st.columns(3)
            for i, movie in enumerate(top_movies):
                with cols[i % 3]:
                    st.write(f"**👉 {movie['title']}**")
                    st.markdown(f"⭐ **Rating:** {movie['vote_average']} / 10 ({movie['vote_count']} votes)")
                    st.markdown(f"📖 **Overview:** {movie['overview']}")
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
                    if poster_url:
                        st.image(poster_url, caption=movie["title"], width=150)
                    else:
                        st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

if __name__ == "__main__":
    main()