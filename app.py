import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load dataset
movies = pd.read_csv("merged_movies.csv")
movies.fillna('', inplace=True)
movies['title_lower'] = movies['title'].str.lower()

# TMDb API setup
TMDB_API_KEY = "887f725faa2dadb468b5baef8c697023"

# Function to fetch movie rating & poster
def fetch_movie_details(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response.get('results'):
        movie_data = response['results'][0]
        return movie_data.get('vote_average', 'N/A'), movie_data.get('poster_path', None)
    return 'N/A', None

# Find closest matching title
def find_closest_match(title):
    matches = get_close_matches(title.lower(), movies['title_lower'], n=1, cutoff=0.6)
    return matches[0] if matches else None

# TF-IDF setup
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Overview-based recommendations
def overview_based_recommendations(title):
    matched_title = find_closest_match(title)
    if not matched_title:
        return pd.DataFrame()
    idx = movies[movies['title_lower'] == matched_title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Genre-based recommendations
def genre_based_recommendations(title):
    matched_title = find_closest_match(title)
    if not matched_title:
        return pd.DataFrame()
    genre = movies[movies['title_lower'] == matched_title]['genres'].values[0]
    return movies[movies['genres'] == genre].sort_values(by='popularity', ascending=False).head(5)

# Top-rated movies from TMDb
def top_rated_movies():
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}&language=en-US&page=1"
    response = requests.get(url).json()
    top_movies = []
    if response.get('results'):
        for item in response['results'][:5]:
            top_movies.append({
                'title': item['title'],
                'overview': item['overview'],
                'rating': item.get('vote_average', 'N/A'),
                'poster_path': item.get('poster_path')
            })
    return top_movies

# Streamlit UI
st.set_page_config(page_title="AI Movie Recommender System", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #111; color: white; }
        .title-style { font-size: 40px; font-weight: bold; color: #f4c10f; }
        .section { border-bottom: 2px solid #444; padding: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-style">🎬 AI Movie Recommender System</div>', unsafe_allow_html=True)

movie_input = st.text_input("🎥 Enter a movie name:")

if st.button("🚀 Recommend"):
    if movie_input:
        matched_movie = find_closest_match(movie_input)
        if matched_movie:
            st.subheader(f"🎯 Recommendations for: **{matched_movie.title()}**")

            st.subheader("📘 Based on Overview")
            overview_recs = overview_based_recommendations(matched_movie)
            if not overview_recs.empty:
                for _, row in overview_recs.iterrows():
                    st.markdown(f"**🎬 {row['title']}**")
                    st.markdown(f"📘 Overview: {row.get('overview', '⚠ No overview available.')}")
                    rating, poster_url = fetch_movie_details(row['title'])
                    st.markdown(f"⭐ Rating: {rating}/10")
                    if poster_url:
                        st.image(f"https://image.tmdb.org/t/p/w200{poster_url}", width=120)
                    else:
                        st.markdown("🖼 No poster available.")
                    st.markdown("---")
            else:
                st.markdown("⚠ No overview-based recommendations available.")

            st.subheader("🎭 Based on Genre")
            genre_recs = genre_based_recommendations(matched_movie)
            if not genre_recs.empty:
                for _, row in genre_recs.iterrows():
                    st.markdown(f"**🎬 {row['title']}**")
                    st.markdown(f"📘 Overview: {row.get('overview', '⚠ No overview available.')}")
                    rating, poster_url = fetch_movie_details(row['title'])
                    st.markdown(f"⭐ Rating: {rating}/10")
                    if poster_url:
                        st.image(f"https://image.tmdb.org/t/p/w200{poster_url}", width=120)
                    else:
                        st.markdown("🖼 No poster available.")
                    st.markdown("---")
            else:
                st.markdown("⚠ No genre-based recommendations available.")

            st.subheader("⭐ Top Rated Movies (TMDb)")
            top_rated = top_rated_movies()
            for movie in top_rated:
                st.markdown(f"**🎬 {movie['title']}**")
                st.markdown(f"📘 Overview: {movie.get('overview', '⚠ No overview available.')}")
                st.markdown(f"⭐ Rating: {movie['rating']}/10")
                if movie.get('poster_path'):
                    st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}", width=120)
                else:
                    st.markdown("🖼 No poster available.")
                st.markdown("---")
        else:
            st.error("❌ Movie not found. Try another name!")
    else:
        st.error("❌ Please enter a valid movie name.")
