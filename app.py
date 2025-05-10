import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movie dataset
movies = pd.read_csv("merged_movies.csv")
movies.fillna('', inplace=True)

# TMDb API setup
TMDB_API_KEY = '887f725faa2dadb468b5baef8c697023'

# Function to fetch movie rating from TMDb
def fetch_movie_rating(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response['results']:
        return response['results'][0].get('vote_average', 'N/A')
    return 'N/A'

# TF-IDF vectorizer for overview-based recommendations
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['overview'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Genre-based movie lookup
def genre_based_recommendations(title):
    if title not in movies['title'].values:
        return []
    genre = movies[movies['title'] == title]['genres'].values[0]
    return movies[movies['genres'] == genre].sort_values(by='popularity', ascending=False).head(5)

# Overview-based recommendation
def overview_based_recommendations(title):
    if title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Rating-based top movies
def top_rated_movies():
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}&language=en-US&page=1"
    response = requests.get(url).json()
    top_movies = []
    if 'results' in response:
        for item in response['results'][:5]:
            top_movies.append({
                'title': item['title'],
                'overview': item['overview'],
                'rating': item['vote_average'],
                'poster_path': item['poster_path']
            })
    return top_movies

# Streamlit UI
st.set_page_config(page_title="AI Movie Recommender System", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #111;
            color: white;
        }
        .title-style {
            font-size: 40px;
            font-weight: bold;
            color: #f4c10f;
        }
        .section {
            border-bottom: 2px solid #444;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-style">🎬 AI Movie Recommender System</div>', unsafe_allow_html=True)

movie_input = st.text_input("🎥 Enter a movie name:")

if st.button("🚀 Recommend"):
    if movie_input:
        st.subheader("🎯 Based on Overview")
        overview_recs = overview_based_recommendations(movie_input.title())
        for _, row in overview_recs.iterrows():
            st.markdown(f"**🎬 {row['title']}**")
            st.markdown(f"**📘 Overview:** {row['overview']}")
            st.markdown(f"⭐ Rating: {fetch_movie_rating(row['title'])}/10")
            st.image(f"https://image.tmdb.org/t/p/w200{row['poster_path']}", width=120)
            st.markdown("---")

        st.subheader("🎭 Based on Genre")
        genre_recs = genre_based_recommendations(movie_input.title())
        for _, row in genre_recs.iterrows():
            st.markdown(f"**🎬 {row['title']}**")
            st.markdown(f"**📘 Overview:** {row['overview']}")
            st.markdown(f"⭐ Rating: {fetch_movie_rating(row['title'])}/10")
            st.image(f"https://image.tmdb.org/t/p/w200{row['poster_path']}", width=120)
            st.markdown("---")

        st.subheader("⭐ Top Rated Movies (TMDb)")
        top_rated = top_rated_movies()
        for movie in top_rated:
            st.markdown(f"**🎬 {movie['title']}**")
            st.markdown(f"**📘 Overview:** {movie['overview']}")
            st.markdown(f"⭐ Rating: {movie['rating']}/10")
            st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}", width=120)
            st.markdown("---")
    else:
        st.error("❌ Please enter a valid movie name.")
